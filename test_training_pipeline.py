"""End-to-end tests of selection, inference export and checkpoint seed provenance."""
import unittest,tempfile,subprocess,sys,json
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parent
class TrainingPipeline(unittest.TestCase):
 def test_training_reload_and_seed_metadata(self):
  config=json.loads((ROOT/'test-model-config.json').read_text());kind=config.pop('_fixture')
  with tempfile.TemporaryDirectory() as tmp:
   tmp=Path(tmp);rng=np.random.default_rng(4);data={}
   for split,n in [('train',20),('val',8),('test',8)]:
    if kind=='tokens':x=rng.integers(1,config['vocabulary'],(n,8),dtype=np.int64)
    elif kind=='risk':x=rng.normal(size=(n,config['features'])).astype(np.float32)
    else:x=rng.normal(size=(n,*kind)).astype(np.float32)
    data['x_'+split]=x;data['y_'+split]=np.arange(n)%config['classes']
   np.savez_compressed(tmp/'data.npz',**data);(tmp/'config.json').write_text(json.dumps(config))
   args=[sys.executable,str(ROOT/'train.py'),'--data',str(tmp/'data.npz'),'--config',str(tmp/'config.json'),'--batch-size','8','--epochs','2','--seed','11','--precision-floor','0']
   trained=tmp/'trained';r=subprocess.run(args+['--output',str(trained)],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
   m=json.loads((trained/'metrics.json').read_text());history=json.loads((trained/'history.json').read_text());self.assertEqual(m['selected_epoch'],min(history,key=lambda row:row['validation_loss'])['epoch']);self.assertTrue((trained/'inference.pt').exists());self.assertEqual(m['test']['rows'],8)
   ck=torch.load(trained/'inference.pt',map_location='cpu',weights_only=True);self.assertEqual(ck['seed'],11)
   evaluated=tmp/'evaluated';r=subprocess.run(args+['--output',str(evaluated),'--evaluate-only','--checkpoint',str(trained/'best.pt'),'--seed','999'],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
   again=json.loads((evaluated/'metrics.json').read_text());self.assertEqual(again['seed'],11);self.assertEqual(again['test'],m['test'])
if __name__=='__main__':unittest.main()
