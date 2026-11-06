from pathlib import Path
source=Path(__file__).with_name('compare_overrides.py').read_text()
source=source.split('for dtype in ')[0] + source[source.index('torch.use_deterministic_algorithms(True)'):]
exec(compile(source,__file__,'exec'),globals())
