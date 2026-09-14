"""Create an explicitly synthetic fixture for the CPU evaluation walkthrough."""
import argparse
from pathlib import Path

import numpy as np

from eyeinhand.artifacts import atomic_json
from eyeinhand.mesh import sample_surface

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
v=np.array([[0.,0,0],[.2,0,0],[0,.2,0],[0,0,.2]])
f=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
np.savez_compressed(a.output/'mesh.npz',vertices=v,faces=f)
np.savez_compressed(a.output/'prediction.npz',points=sample_surface(v,f,count=1000,seed=0))
np.save(a.output/'transform.npy',np.eye(4))
atomic_json(a.output/'README.json',{'synthetic_fixture':True,'measurement_run':False,'units':'meters'})
