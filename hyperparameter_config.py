from pydantic import RootModel, BaseModel, Field
from typing import List, Optional, Any, Dict

from envelopes.envelope_config import EnvelopeConfig

class HParamStructure(BaseModel):
    latent_dim: Optional[int]

    def to_string(self) -> List[str]:
        """
        Returns a list of compact string parts for each non-None field, using abbreviations.
        """
        parts = []
        for name, val in self.model_dump(exclude_none=True).items():
            abbrev = HParamConfig.get_abbrev(name)
            parts.append(f"{abbrev}={val}")
        return parts

    @classmethod
    def from_string(cls, parts: List[str]) -> "HParamStructure":
        """
        Constructs a HParamStructure from a list of abbreviation=value strings.
        """
        struct_kwargs: Dict[str, Any] = {}
        # Determine the abbreviation for latent_dim
        latent_abbrev = HParamConfig.get_abbrev("latent_dim")
        for part in parts:
            if "=" not in part:
                continue
            key, val = part.split("=", 1)
            if key == latent_abbrev:
                struct_kwargs["latent_dim"] = int(val)
        return cls(**struct_kwargs)

class HParamConfig(BaseModel):
    structure: Optional[HParamStructure]
    seed: Optional[int]
    envelopes: Dict[str, EnvelopeConfig] = Field(default_factory=dict)

    def envelope_summary(self, exclude: Optional[List[str]] = None) -> str:
        """
        Returns a compact human-readable string summarizing all envelopes.
        Example: "classification=linear[T=0.5,slope=0.1,ss=0.05]"
        If step_size is present, it is included as ss=... in the summary.
        """
        parts = []
        for name, env in sorted(self.envelopes.items()):
            parts.append(f"{name}={env.to_slug(exclude=exclude)}]")
        return "; ".join(parts)
    
    # Core field abbreviations
    __abbrev_map__ = {
        'seed':     's',
        'latent_dim':'d',
    }

    @classmethod
    def _generate_slug(cls, key: str) -> str:
        # Create a short slug from the key, e.g. 'free_nats' -> 'fn'
        parts = key.split('_')
        slug = ''.join(p[0] for p in parts if p)
        slug = slug[:4]
        # Avoid collisions with existing abbreviations
        existing = set(cls.__abbrev_map__.values())
        base = slug
        idx = 1
        while slug in existing:
            slug = f"{base}{idx}"
            idx += 1
        return slug

    @classmethod
    def get_abbrev(cls, key: str) -> str:
        """
        Return the abbreviation for a given key, generating a new one for extras.
        """
        if key in cls.__abbrev_map__:
            return cls.__abbrev_map__[key]
        # Generate and cache a new slug for this extra key
        slug = cls._generate_slug(key)
        cls.__abbrev_map__[key] = slug
        return slug

    def to_string(self, exclude_seed: bool = False) -> str:
        parts = []
        if not exclude_seed:
            parts.append(f"s={self.seed}")
        if self.structure:
            parts.extend(self.structure.to_string())
        # Serialize envelopes as <envname>=<envelope_slug>
        for env_name, envelope in sorted(self.envelopes.items()):
            parts.append(f"{env_name}={envelope.to_slug()}")
        return "_".join(parts)

    @classmethod
    def from_string(cls, s: str) -> "HParamConfig":
        seed: Optional[int] = None
        struct_kwargs: Dict[str, Any] = {}
        envs: Dict[str, EnvelopeConfig] = {}

        for part in s.split("_"):
            if "=" not in part:
                continue
            key, val = part.split("=", 1)
            if key == "s":
                seed = int(val)
            elif key == "d":
                struct_kwargs['latent_dim'] = int(val)
            else:
                envs[key] = EnvelopeConfig.from_slug(val)
        structure = HParamStructure(**struct_kwargs) if struct_kwargs else None
        return cls(
            structure=structure,
            seed=seed,
            envelopes=envs
        )

class HParamSweep(RootModel[dict[str, Any]]):
    def expand(self) -> List[HParamConfig]:
        """
        Expand the sweep into a list of HParamConfig instances.
        Supports:
        - 'seeds' as an int (number of seeds) or list of ints.
        - Scalars or lists for 'latent_dim', etc.
        - Nested sweeps inside envelope 'parameters' dict.
        - Experiment-specific sweep parameters (e.g., 'latent_dim') live under 'structure'.
        """
        # Use root payload as the sweep dictionary
        sweep_dict = self.root

        def listify(val):
            if val is None:
                return [None]
            return val if isinstance(val, list) else [val]

        def recursive_expand(d: dict) -> List[dict]:
            """
            Recursively expands a dictionary with possible sweep shorthand.
            Example:
            {"T": {"loss_pct": [0.1, 0.5]}, "slope": {"loss_pct": [0.01, 0.1]}}
            becomes 4 separate dicts with all combinations, preserving the nested structure.
            """
            keys = list(d.keys())
            if not keys:
                return [{}]

            first_key = keys[0]
            rest = keys[1:]

            first_val = d[first_key]
            if isinstance(first_val, dict) and ("loss_pct" in first_val):
                subkey, subvals = list(first_val.items())[0]
                expanded = []
                for val in subvals:
                    rest_combos = recursive_expand({k: d[k] for k in rest})
                    for rest_combo in rest_combos:
                        combo = {first_key: {subkey: val}, **rest_combo}
                        expanded.append(combo)
                return expanded
            else:
                rest_combos = recursive_expand({k: d[k] for k in rest})
                return [{first_key: first_val, **rest_combo} for rest_combo in rest_combos]

        seeds_field = sweep_dict.get("seeds", 0)
        if isinstance(seeds_field, int):
            seeds = list(range(seeds_field))
        elif isinstance(seeds_field, list):
            seeds = seeds_field
        else:
            raise ValueError("'seeds' must be int or list of ints")
        structure = sweep_dict.get("structure", {})
        latent_dims = listify(structure.get("latent_dim", None))
        envelopes = sweep_dict.get("envelopes", {})

        # Build envelope configurations per envelope name
        envelope_expansions = {}
        for env_name, env_list in envelopes.items():
            expanded_envs = []
            for env_cfg in env_list:
                shape = env_cfg.get("shape", "identity")
                params = env_cfg.get("parameters", {})
                step_sizes = env_cfg.get("step_size", 0.05)
                step_sizes = step_sizes if isinstance(step_sizes, list) else [step_sizes]

                param_combos = recursive_expand(params) if params else [{}]
                for step_size in step_sizes:
                    for combo in param_combos:
                        expanded_envs.append(EnvelopeConfig(shape=shape, parameters=combo, step_size=step_size))

            envelope_expansions[env_name] = expanded_envs or [None]

        # Cartesian product over all fields
        configs = []
        for seed in seeds:
            for latent_dim in latent_dims:
                envelope_names = list(envelope_expansions.keys())
                envelope_lists = [envelope_expansions[name] for name in envelope_names]

                def product(lists, prefix=[]):
                    if not lists:
                        yield prefix
                        return
                    for item in lists[0]:
                        yield from product(lists[1:], prefix + [item])

                for combo in product(envelope_lists):
                    env_dict = {
                        name: env for name, env in zip(envelope_names, combo) if env is not None
                    }
                    configs.append(HParamConfig(
                        structure=HParamStructure(latent_dim=latent_dim),
                        seed=seed,
                        envelopes=env_dict
                    ))
        return configs