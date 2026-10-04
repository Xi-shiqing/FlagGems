"""Switch the six registered function bodies between baseline and candidate."""
import ast
import importlib
from pathlib import Path


def controls(baseline):
    result = []
    for name in ('mm', 'addmm', 'bmm'):
        module = importlib.import_module('flag_gems.ops.'+name)
        path = Path(baseline)/('flag_gems/ops/'+name+'.py')
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in (name, name+'_out')]
        assert len(nodes) == 2
        namespace = dict(vars(module))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        for node in nodes:
            function = getattr(module, node.name)
            assert function.__closure__ is None
            result.append((function, namespace[node.name].__code__, function.__code__))
    return result


def switch(entries, candidate):
    for function, baseline_code, candidate_code in entries:
        function.__code__ = candidate_code if candidate else baseline_code
