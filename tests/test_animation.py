import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from primal_mapf.animation import render_animation, sampled_steps, validate_record
from primal_mapf.provenance import file_sha256


class AnimationTests(unittest.TestCase):
    def record(self):
        return dict(map_name='maze',variant='improved',checkpoint='checkpoints/fine_tuned/improved/maze.pt',
                    agents=2,seed=123,execution='sequential',environment_profile='historical',
                    obstacles=[[0]*4 for _ in range(3)],starts=[(0,0),(2,3)],goals=[(0,2),(2,1)],
                    trace=[[(0,0),(2,3)],[(0,1),(2,2)],[(0,2),(2,1)]],success=True,makespan=2)

    def test_success_and_collision_validation(self):
        record=self.record()
        validate_record(record)
        record['success']=False
        with self.assertRaises(ValueError):
            validate_record(record)
        record=self.record()
        record['trace'][1]=[(0,1),(0,1)]
        with self.assertRaises(ValueError):
            validate_record(record)
        record=self.record()
        record['obstacles'][0][0]=.5
        with self.assertRaises(ValueError):
            validate_record(record)

    def test_frame_sampling_keeps_endpoints(self):
        indices=sampled_steps(513,100)
        self.assertEqual((indices[0],indices[-1]),(0,512))
        self.assertLessEqual(len(indices),100)
        self.assertEqual(sampled_steps(1,2),[0])
        with self.assertRaises(ValueError):
            sampled_steps(10,1)

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'Install the visualization extra to test GIF rendering')
    def test_gif_preserves_trace_and_rejects_overwrite(self):
        from PIL import Image, ImageChops
        record=self.record()
        original=copy.deepcopy(record)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'animation'
            details=render_animation(record,output,max_frames=2)
            self.assertEqual(record,original)
            saved=json.loads((output/'trajectory.json').read_text())
            self.assertEqual(saved['trace'],json.loads(json.dumps(record['trace'])))
            self.assertEqual(details['frame_steps'],[0,2])
            self.assertEqual(details['gif_sha256'],file_sha256(output/'animation.gif'))
            with Image.open(output/'animation.gif') as gif:
                first=gif.convert('RGB')
                self.assertEqual(gif.n_frames,2)
                gif.seek(gif.n_frames-1)
                last=gif.convert('RGB')
                self.assertIsNotNone(ImageChops.difference(first,last).getbbox())
            self.assertTrue((output/'poster.png').exists())
            with self.assertRaises(FileExistsError):
                render_animation(record,output)


if __name__ == '__main__':
    unittest.main()
