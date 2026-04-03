import torch
from pathlib import Path


def clean_state_dict_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.removeprefix("module.")
        new_state_dict[new_key] = v
    return new_state_dict


def main():
    src_dir = Path(".")
    out_dir = src_dir / "cleaned_pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(
        p for p in src_dir.glob("*.pt") if p.is_file() and not p.name.endswith("_cleaned.pt")
    )

    if not pt_files:
        print("No .pt files found in current directory.")
        return

    print(f"Found {len(pt_files)} .pt files")
    for pt_path in pt_files:
        obj = torch.load(pt_path, map_location="cpu")

        # 支持两种格式:
        # 1) 直接是 state_dict
        # 2) checkpoint 字典中有 state_dict 字段
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj["state_dict"] = clean_state_dict_keys(obj["state_dict"])
            cleaned_obj = obj
        elif isinstance(obj, dict):
            cleaned_obj = clean_state_dict_keys(obj)
        else:
            print(f"[skip] {pt_path.name}: unsupported top-level type {type(obj)}")
            continue

        out_path = out_dir / f"{pt_path.stem}_cleaned.pt"
        torch.save(cleaned_obj, out_path)
        print(f"[ok] {pt_path.name} -> {out_path}")


if __name__ == "__main__":
    main()