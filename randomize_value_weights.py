import torch


def randomize(values: torch.Tensor) -> torch.Tensor:
    flattened = values.flatten()
    perm = torch.randperm(flattened.size(0))
    shuf_flat = flattened[perm]
    shape = values.shape
    return shuf_flat.reshape(shape)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--weights",
            type=str,
            default=None,
            help="Path to saved weights."
    )
    parser.add_argument(
            "--save_path",
            type=str,
            default=None,
            help="Path to save randomized weights."
    )

    args = parser.parse_args()
    state_dict = torch.load(args.weights)
    orig_vals = state_dict["value"]
    shuffled_vals = randomize(orig_vals)
    
    state_dict["value"] = shuffled_vals
    torch.save(state_dict, args.save_path)
