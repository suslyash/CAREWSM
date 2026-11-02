"""Small TASK-004 registry/YAML/forward-backward smoke."""
from pathlib import Path
import sys, torch, yaml
sys.path.insert(0, str(Path(__file__).parents[1]))
from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
import chimera_plugin
from src.audio.loss import MaskedMultitaskBCE

def main():
    chimera_plugin.register(); assert {"carewsm_tscmamba","carewsm_mptsnet","carewsm_md_former"} <= set(MODELS.keys())
    config=yaml.safe_load(Path("configs/audio/task004_models.yaml").read_text())
    for spec in config["models"]:
        try: MODELS.create(spec["name"], **spec["params"])
        except ModuleNotFoundError: pass
    x=torch.randn(4,9,8); mask=torch.tensor([[1]*9,[1]*6+[0]*3,[1]*2+[0]*7,[0]*9],dtype=torch.bool); targets=torch.tensor([[1,0],[0,1],[1,0],[0,1]]); observed=torch.tensor([[1,1],[1,0],[0,1],[0,0]],dtype=torch.bool)
    batch=Batch(inputs={"native":x},targets=targets,masks={"sequence_mask":mask})
    for name in ("carewsm_mptsnet","carewsm_md_former","carewsm_pooled_mlp"):
        params={"input_dim":8,"hidden_dim":16}; params.update({"num_heads":4} if name == "carewsm_md_former" else {}); model=MODELS.create(name,**params); out=model(batch); loss=MaskedMultitaskBCE.compute(out.preds,targets,observed); loss.backward(); assert torch.isfinite(out.preds).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()); print(name, tuple(out.preds.shape), float(loss.detach()))
    try: MODELS.create("carewsm_tscmamba",input_dim=8,hidden_dim=16,mamba_required=True)
    except ModuleNotFoundError as e: print("carewsm_tscmamba BLOCKED:", e)
if __name__ == "__main__": main()
