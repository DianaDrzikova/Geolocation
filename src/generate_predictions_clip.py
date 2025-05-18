import argparse, os, pathlib, csv, torch
from PIL import Image
import open_clip
from geoclip import GeoCLIP
from tqdm import tqdm
from heads import CLIPRegressorBasic, CLIPRegressorGeoClip

def eval(args):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.model_type == "basic":
        gc = GeoCLIP()
        _, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14")
        gc.requires_grad_(False)
        gc.eval()

        print(f"CLIP backbone: {gc.image_encoder.__class__.__name__}")

        model = CLIPRegressorGeoClip(gc).to(device)
        model.gc.half() 
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=False)

    if args.model_type == "basic":
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k", device=device)
        clip_model.eval(); clip_model.requires_grad_(False)

        model = CLIPRegressorBasic(clip_model).to(device)
        model.clip.half()  
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=False)

    img_dir = pathlib.Path(
        args.img_dir if args.img_dir else os.path.join(os.environ["SCRATCHDIR"], "query_photos")
    )
    paths = sorted([p for p in img_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    print("Found %d images in %s" % (len(paths), img_dir))
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "latitude_radians", "longitude_radians"])
        for p in tqdm(paths, desc="Evaluating images"):
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).half().to(device)
            with torch.no_grad():
                pred_rad = model(img)[0].cpu()    
            writer.writerow([p.name, pred_rad[0].item(), pred_rad[1].item()])

    print(f"Saved {len(paths)} predictions to {args.output_csv}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="Inference with trained CLIP regressor → CSV (lat/lon in radians)")
    parser.add_argument("--ckpt", required=True, help="checkpoint .pt saved during training")
    parser.add_argument("--img_dir", required=False, default=None, help="directory with query images (default $SCRATCHDIR/query_photos)")
    parser.add_argument("--output_csv", required=True, help="where to save predictions")
    parser.add_argument("--model_type", type=str, default="basic", choices=["basic", "geoclip"])

    args = parser.parse_args()

    eval(args)