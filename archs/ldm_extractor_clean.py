from functools import partial
from collections import defaultdict
import logging
import math
import torch
import torch.nn.functional as F
from torch import nn


class LDMExtractor:

    def __init__(self, cfg, pipe):

        self.device = pipe.device

        self.unet = pipe.unet
        self.vae = pipe.vae
        self.clip = pipe.text_encoder
        self.clip_tokenizer = pipe.tokenizer

        self.t_init = 100     #  100, 261 a fixed value for SatDIFT
        self.timestep = nn.Parameter(
            torch.tensor(float(self.t_init), requires_grad=True), requires_grad=True
        )

        self.logger = logging.getLogger()

        self.generator = torch.Generator(self.device).manual_seed(42)
        self.batchsize = cfg.batch_size
        self.resize_outputs = cfg.get("resize_outputs", -1)

        if self.resize_outputs > 0:
            self.pyramid_output = False
        else:
            self.pyramid_output = True
            scales = (8, 16, 32, 64)
            self.feat_size_scale = [cfg.img_size // s for s in scales]

        self.prompt = cfg.get("prompt", "A satellite image")
        self.negative_prompt = cfg.get("negative_prompt", "")

        self.change_cond(self.prompt, self.batchsize, "cond")
        self.change_cond(self.negative_prompt, self.batchsize, "uncond")

        self.latent_res = cfg.img_size // 8

        self.layer_idxs = cfg.get("layer_idxs", None)

        # set up hooks and collect layers to extract features
        if self.layer_idxs:
            self.collected_layers, self.collect_layer_names = self.collect_layers(self.unet)
            self.collected_dims = self.collect_layer_dims(self.collected_layers, self.collect_layer_names)
            self.register_layer_hooks(self.collected_layers)

        self.reset_feats()

    def register_layer_hooks(self, modules):

        def hook_dif_mod(mod, input, output):
            mod.feats = output.detach()

        for i, module in enumerate(modules):
            module.register_forward_hook(partial(hook_dif_mod))

        self.logger.info("Hooks registered")

    def change_cond(self, prompts, batch_size, cond_type="cond"):
        with torch.no_grad():
            with torch.autocast("cuda"):
                _, new_cond = get_tokens_embedding(self.clip_tokenizer, self.clip, self.device, prompts)

                if type(prompts) == str:
                    new_cond = new_cond.expand((batch_size, *new_cond.shape[1:]))
                elif type(prompts) == list and len(prompts) == batch_size:
                    pass
                else:
                    raise ValueError("Check the prompt type and batch size")

                new_cond = new_cond.to(self.device)
                if cond_type == "cond":
                    self.cond = new_cond
                    self.prompt = prompts
                elif cond_type == "uncond":
                    self.uncond = new_cond
                    self.negative_prompt = prompts
                else:
                    raise NotImplementedError

    def collect_layers(self, unet):
        layers = []
        layer_names = []
        for block_name, block_module in self.layer_idxs.items():
            blocks = getattr(unet, block_name, None)

            if blocks is None:
                raise ValueError(f"Block name {block_name} not found in unet")
            if not isinstance(blocks, torch.nn.ModuleList):
                blocks = [blocks]

            for i in range(len(blocks)):
                for mod_name, mod_idx in block_module.items():

                    unet_modules = get_named_modules_with_suffix(blocks[i], mod_name)
                    for j in range(len(unet_modules)):
                        if mod_idx == "all" or (i, j) in mod_idx:
                            self.logger.info(f"Collect layer {block_name} {i} {mod_name} {j}")
                            layers.append(unet_modules[j])
                            layer_names.append(f"{block_name}_{i}_{mod_name}_{j}")

        self.logger.info(f"The number of collected layers: {len(layers)}")
        return layers, layer_names

    def collect_layer_dims(self, modules, module_names):

        if self.pyramid_output:
            dim_groups = {}
            # run a dummmy forward pass to get the output dimensions
            dummy_input = torch.zeros(1, 4, self.latent_res, self.latent_res).to(self.device)
            dummy_cond = torch.zeros((1, self.cond.shape[1], self.cond.shape[2])).to(self.device)

            def hook_dims(module, input, output):
                module.output_shape = output.shape

            hook_handles = []
            for layer, name in zip(modules, module_names):
                hook_handles.append(layer.register_forward_hook(partial(hook_dims)))

            _ = self.unet(dummy_input, 1, dummy_cond)
            for layer in modules:
                if len(layer.output_shape) == 4:
                    b, c, w, _ = layer.output_shape
                elif len(layer.output_shape) == 3:
                    b, l, c = layer.output_shape
                    w = int(math.sqrt(l))
                else:
                    raise ValueError(f"Layer shape {layer.output_shape} not supported")
                if w not in dim_groups:
                    dim_groups[w] = []
                dim_groups[w].append(c)

            # delete the specific hooks
            for handle in hook_handles:
                handle.remove()
            return dim_groups

        else:
            dim_list = []
            for layer, name in zip(modules, module_names):
                if "resnet" in name:
                    dim_list.append(layer.time_emb_proj.out_features)
                elif "conv" in name:
                    dim_list.append(layer.out_channels)
                elif "attn" in name:
                    dim_list.append(layer.to_out[0].out_features)
                else:
                    raise ValueError(f"Layer type {name} not supported, cannot determine layer dimension")
            return dim_list

    def collect_feats(self, layers):
        # collect features from the layers at a specific timestep
        feats = []

        for module in layers:
            module_feats = module.feats
            # TODO: here is tailored for self-attn/cross-attn outputs, need to generalize or check if it's true for other cases
            if len(module_feats.shape) == 3:
                w = int(math.sqrt(module_feats.shape[1]))
                module_feats = module_feats.reshape(module_feats.shape[0], w, w, module_feats.shape[2]).permute(0, 3, 1,
                                                                                                                2)

            if self.resize_outputs > 0:
                module_feats = F.interpolate(module_feats, size=self.resize_outputs, mode="bilinear")

            feats.append(module_feats)

        if not self.pyramid_output:  # self.resize_outputs > 0
            feats = torch.cat(feats,
                              dim=1)  # can be concatenated along the channel dimension since the spatial dimensions are the same
        else:
            feat_group = defaultdict(list)
            for i, f in enumerate(feats):
                w = f.shape[-1]
                feat_group[w].append(f)
            feats = [torch.cat(feat_group[w], dim=1) for w in sorted(feat_group.keys(), reverse=True)]

        return feats

    def reset_feats(self):
        if hasattr(self, "collected_layers"):
            for module in self.collected_layers:
                module.feats = None
            self.feats = {}

    def forward(self, latents, prompts=None, guidance_scale=-1):
        # clear the features
        torch.cuda.empty_cache()
        bs = latents.shape[0]

        if prompts is None:
            prompts = self.prompt
            if bs != self.batchsize:
                self.batchsize = bs
                self.change_cond(self.prompt, bs, "cond")
                self.change_cond(self.negative_prompt, bs, "uncond")
        else:
            self.change_cond(prompts, bs, "cond")
            self.change_cond(self.negative_prompt, bs, "uncond")

        self.reset_feats()
        with torch.no_grad():
            # print("************", latents.shape, torch.max(latents), torch.min(latents), torch.mean(latents))
            _ = self.unet(latents, (torch.ones(bs) * self.timestep).to(latents.device), encoder_hidden_states=self.cond)
            # _ = self.unet(latents, self.t_init, encoder_hidden_states=self.cond)
            self.feats = self.collect_feats(self.collected_layers)

        output_feats = self.feats
        return output_feats, None


def get_tokens_embedding(clip_tokenizer, clip, device, prompt):
    tokens = clip_tokenizer(
        prompt,
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    )
    input_ids = tokens.input_ids.to(device)

    embedding = clip(input_ids)[0]
    return tokens, embedding


def get_xt_next(xt, et, at, at_next, eta):
    """
    Uses the DDIM formulation for sampling xt_next
    Denoising Diffusion Implicit Models (Song et. al., ICLR 2021).
    """
    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
    if eta == 0:
        c1 = 0
    else:
        c1 = (
                eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        )
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(et) + c2 * et
    return x0_t, xt_next


def get_named_modules_with_suffix(module, suffix):
    """Helper function to retrieve modules with names ending in a specific suffix."""
    collected_modules = []
    for n, m in module.named_modules():
        if n.endswith(suffix):
            if isinstance(m, torch.nn.ModuleList):
                for i in range(len(m)):
                    collected_modules.append(m[i])
            else:
                collected_modules.append(m)
    return collected_modules

