import ast


def convert_lecture_py(source: str) -> str:
    """Walk the AST of a CS336 lecture .py file and emit the prose inside text(...) calls
    as a markdown document, preserving order. Ignore image(...), link(...), and other calls."""
    tree = ast.parse(source)
    lines: list[str] = []

    class TextCollector(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name == "text" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    lines.append(arg.value)
            self.generic_visit(node)

    TextCollector().visit(tree)
    return "\n\n".join(lines).strip() + "\n"
