from pathlib import Path
import sys,runpy
script=Path(__file__).with_name('queue_training_layout.py')
sys.argv=[str(script),'--precision','fp32']
runpy.run_path(str(script),run_name='__main__')
