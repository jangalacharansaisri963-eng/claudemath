"""Interactive command-line interface for claudemath."""

from __future__ import annotations

import ast
import importlib
import json
import math
import operator
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable


VERSION = "0.1"

DOMAINS = [
    "algebra",
    "arithmetic",
    "calculus",
    "combinatorics",
    "complex_numbers",
    "coordinate_systems",
    "differential_equations",
    "discrete_mathematics",
    "fourier_analysis",
    "geometry",
    "graph_theory",
    "linear_algebra",
    "matrix_operations",
    "number_theory",
    "numerical_analysis",
    "polynomial_operations",
    "probability",
    "statistics",
    "trigonometry",
    "vector_operations",
]


# ----------------------------------------------------
# Angle mode: lets `sin(30)`, `asin(x)`, etc. be typed
# directly in degrees/gradians without a _deg/_rad suffix.
# ----------------------------------------------------

ANGLE_MODES = ("radian", "degree", "gradian")

ANGLE_MODE_ALIASES = {
    "radian": "radian",
    "radians": "radian",
    "rad": "radian",
    "rads": "radian",
    "degree": "degree",
    "degrees": "degree",
    "deg": "degree",
    "degs": "degree",
    "gradian": "gradian",
    "gradians": "gradian",
    "gradiant": "gradian",
    "gradiants": "gradian",
    "grad": "gradian",
    "grads": "gradian",
    "gon": "gradian",
    "gons": "gradian",
}

# Trig functions whose FIRST argument is an angle (convert it TO radians
# before calling, since the underlying implementations are radian-based).
ANGLE_INPUT_FUNCTIONS = {"sin", "cos", "tan", "cot", "sec", "csc"}

# Trig functions whose RESULT is an angle (convert it FROM radians after
# calling).
ANGLE_OUTPUT_FUNCTIONS = {
    "asin", "acos", "atan", "acot", "asec", "acsc", "atan2",
}


def normalize_angle_mode(mode: str) -> str:
    """Resolve a user-typed mode name (with aliases/typos) to a canonical mode."""
    key = mode.strip().lower()

    if key not in ANGLE_MODE_ALIASES:
        raise ValueError(
            f"Unknown angle mode: {mode!r}. "
            "Use one of: degree, radian, gradian"
        )

    return ANGLE_MODE_ALIASES[key]


def angle_to_radians(value: float, mode: str) -> float:
    if mode == "degree":
        return math.radians(value)

    if mode == "gradian":
        return value * math.pi / 200.0

    return value


def angle_from_radians(value: float, mode: str) -> float:
    if mode == "degree":
        return math.degrees(value)

    if mode == "gradian":
        return value * 200.0 / math.pi

    return value


def _config_path() -> str:
    return os.path.join(
        os.path.expanduser("~"),
        ".claudemath",
        "config.json",
    )


def load_saved_angle_mode() -> str:
    """Read the persisted default angle mode, falling back to 'radian'."""
    try:
        with open(_config_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)

        return normalize_angle_mode(data.get("angle_mode", "radian"))
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
        return "radian"


def save_angle_mode(mode: str) -> None:
    """Persist the default angle mode so it applies to future sessions too."""
    path = _config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
        data = {}

    data["angle_mode"] = mode

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    RED = "\033[91m"


def color(text: str, value: str) -> str:
    return f"{value}{text}{Colors.RESET}"


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_banner() -> None:
    c = Colors

    print()
    print(color("╭──────────────────────────────────────────────────────╮", c.CYAN))
    print(
        color("│", c.CYAN)
        + color(
            "                 claudemath                         ",
            c.BOLD + c.WHITE,
        )
        + color("│", c.CYAN)
    )
    print(
        color("│", c.CYAN)
        + color(
            "        Advanced Pure-Python Mathematics              ",
            c.MAGENTA,
        )
        + color("│", c.CYAN)
    )
    print(
        color("│", c.CYAN)
        + color("                                                      ", c.WHITE)
        + color("│", c.CYAN)
    )
    print(
        color("│", c.CYAN)
        + color(f"        v{VERSION:<8}", c.GREEN)
        + color(f"{len(DOMAINS)} domains", c.YELLOW)
        + color(" " * 23, c.WHITE)
        + color("│", c.CYAN)
    )
    print(color("╰──────────────────────────────────────────────────────╯", c.CYAN))
    print()
    print(color("  Mathematics engine ready.", c.GREEN))
    print(
        color("  Use ", c.GRAY)
        + color("solve function(...)", c.CYAN)
        + color(" to evaluate mathematics.", c.GRAY)
    )
    print(
        color("  Type ", c.GRAY)
        + color("/help", c.CYAN)
        + color(" for commands.", c.GRAY)
    )
    print()


def print_help() -> None:
    c = Colors

    print()
    print(color("Commands", c.BOLD + c.CYAN))
    print()

    commands = [
        ("/help", "Show available commands"),
        ("/clear", "Clear the terminal"),
        ("/domains", "List mathematics domains"),
        ("/functions", "List available functions"),
        ("/history", "Show session history"),
        ("/mode", "Show or set angle mode (degree/radian/gradian)"),
        ("/version", "Show version"),
        ("/exit", "Exit claudemath"),
    ]

    for command, description in commands:
        print(f"  {color(command, c.GREEN):<30} {description}")

    print()
    print(color("Syntax", c.BOLD + c.MAGENTA))
    print()
    print(f"  {color("solve add(1, 2)", c.WHITE)}")
    print(f"  {color("solve gcd(48, 18)", c.WHITE)}")
    print(f"  {color("solve factorial(10)", c.WHITE)}")
    print(f"  {color("solve sqrt(25)", c.WHITE)}")
    print(f"  {color("solve add(2, multiply(3, 4))", c.WHITE)}")
    print(f"  {color("solve sin(pi / 2)", c.WHITE)}")
    print()
    print(color("Angle mode", c.BOLD + c.MAGENTA))
    print()
    print(f"  {color("mode degree", c.WHITE)}   {color('sin(30) etc. now read/return degrees', c.GRAY)}")
    print(f"  {color("mode radian", c.WHITE)}   {color('default — sin(30) means 30 radians', c.GRAY)}")
    print(f"  {color("mode gradian", c.WHITE)}  {color('sin(30) etc. now read/return gradians', c.GRAY)}")
    print()


def print_domains() -> None:
    c = Colors

    print()
    print(
        color(
            f"Mathematics domains ({len(DOMAINS)})",
            c.BOLD + c.CYAN,
        )
    )
    print()

    for index, domain in enumerate(DOMAINS, 1):
        print(
            f"  {color(f'{index:02}', c.GRAY)}"
            f"  {color(domain, c.WHITE)}"
        )

    print()


@dataclass(frozen=True)
class FunctionInfo:
    name: str
    domain: str
    function: Callable[..., Any]


class FunctionEngine:
    """Discovers and resolves functions from claudemath domains."""

    def __init__(self) -> None:
        self.functions: dict[str, list[FunctionInfo]] = {}
        self._load_functions()

    def _load_functions(self) -> None:
        for domain_name in DOMAINS:
            try:
                module = importlib.import_module(
                    f"claudemath.{domain_name}"
                )
            except Exception:
                continue

            self._register_module_functions(
                module,
                domain_name,
            )

    def _register_module_functions(
        self,
        module: Any,
        domain_name: str,
    ) -> None:
        for name in dir(module):
            if name.startswith("_"):
                continue

            try:
                value = getattr(module, name)
            except Exception:
                continue

            if not callable(value):
                continue

            info = FunctionInfo(
                name=name,
                domain=domain_name,
                function=value,
            )

            self.functions.setdefault(
                name.lower(),
                [],
            ).append(info)

    def resolve(self, name: str) -> FunctionInfo:
        clean_name = name.strip().lower()

        matches = self.functions.get(clean_name, [])

        if not matches:
            raise NameError(
                f"Unknown function: {name}"
            )

        if len(matches) > 1:
            domains = ", ".join(
                sorted(info.domain for info in matches)
            )

            raise NameError(
                f"Function '{name}' is ambiguous. "
                f"Found in: {domains}"
            )

        return matches[0]

    def all_functions(self) -> list[FunctionInfo]:
        result: list[FunctionInfo] = []

        for matches in self.functions.values():
            result.extend(matches)

        unique: dict[tuple[str, str], FunctionInfo] = {}

        for info in result:
            unique[(info.domain, info.name)] = info

        return sorted(
            unique.values(),
            key=lambda item: (item.domain, item.name),
        )


class ExpressionEvaluator:
    """Safely evaluates values and expressions used by function calls."""

    BINARY_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.MatMult: operator.matmul,
    }

    COMPARISON_OPERATORS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    UNARY_OPERATORS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
    }

    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
        "nan": math.nan,
        "true": True,
        "false": False,
        "none": None,
    }

    BUILTINS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "len": len,
    }

    def __init__(self, engine: FunctionEngine) -> None:
        self.engine = engine
        self.angle_mode: str = load_saved_angle_mode()

    def set_angle_mode(self, mode: str) -> str:
        self.angle_mode = normalize_angle_mode(mode)
        return self.angle_mode

    def evaluate(self, expression: str) -> Any:
        expression = expression.strip()

        if not expression:
            raise ValueError("Empty expression.")

        expression = self._normalize(expression)

        try:
            tree = ast.parse(
                expression,
                mode="eval",
            )
        except SyntaxError as exc:
            raise ValueError(
                f"Invalid expression: {expression}"
            ) from exc

        return self._evaluate_node(tree.body)

    @staticmethod
    def _normalize(expression: str) -> str:
        return expression.replace("^", "**")

    def _evaluate_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(
                node.value,
                (str, int, float, complex, bool),
            ) or node.value is None:
                return node.value

            raise ValueError("Unsupported constant.")

        if isinstance(node, ast.Name):
            name = node.id.lower()

            if name in self.CONSTANTS:
                return self.CONSTANTS[name]

            raise NameError(
                f"Unknown constant or variable: {node.id}"
            )

        if isinstance(node, ast.List):
            return [
                self._evaluate_node(element)
                for element in node.elts
            ]

        if isinstance(node, ast.Tuple):
            return tuple(
                self._evaluate_node(element)
                for element in node.elts
            )

        if isinstance(node, ast.Set):
            return {
                self._evaluate_node(element)
                for element in node.elts
            }

        if isinstance(node, ast.Dict):
            return {
                self._evaluate_node(key): self._evaluate_node(value)
                for key, value in zip(node.keys, node.values)
            }

        if isinstance(node, ast.BinOp):
            operation = self.BINARY_OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError(
                    f"Unsupported operator: "
                    f"{type(node.op).__name__}"
                )

            left = self._evaluate_node(node.left)
            right = self._evaluate_node(node.right)

            return operation(left, right)

        if isinstance(node, ast.UnaryOp):
            operation = self.UNARY_OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError(
                    f"Unsupported unary operator: "
                    f"{type(node.op).__name__}"
                )

            value = self._evaluate_node(node.operand)

            return operation(value)

        if isinstance(node, ast.Compare):
            left = self._evaluate_node(node.left)

            for operator_node, comparator in zip(
                node.ops,
                node.comparators,
            ):
                operation = self.COMPARISON_OPERATORS.get(
                    type(operator_node)
                )

                if operation is None:
                    raise ValueError("Unsupported comparison.")

                right = self._evaluate_node(comparator)

                if not operation(left, right):
                    return False

                left = right

            return True

        if isinstance(node, ast.BoolOp):
            values = [
                self._evaluate_node(value)
                for value in node.values
            ]

            if isinstance(node.op, ast.And):
                return all(values)

            if isinstance(node.op, ast.Or):
                return any(values)

            raise ValueError("Unsupported boolean operator.")

        if isinstance(node, ast.Call):
            return self._evaluate_call(node)

        raise ValueError(
            f"Unsupported expression: "
            f"{type(node).__name__}"
        )

    def _evaluate_call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                "Only direct function calls are supported."
            )

        name = node.func.id

        args = [
            self._evaluate_node(argument)
            for argument in node.args
        ]

        kwargs: dict[str, Any] = {}

        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError(
                    "Dictionary expansion is not supported."
                )

            kwargs[keyword.arg] = self._evaluate_node(
                keyword.value
            )

        builtin = self.BUILTINS.get(name.lower())

        if builtin is not None:
            return builtin(*args, **kwargs)

        info = self.engine.resolve(name)

        if info.domain == "trigonometry" and self.angle_mode != "radian":
            fname = info.name.lower()

            if fname in ANGLE_INPUT_FUNCTIONS and args:
                args = [angle_to_radians(args[0], self.angle_mode), *args[1:]]

            try:
                result = info.function(*args, **kwargs)
            except TypeError as exc:
                raise TypeError(
                    f"{info.domain}.{info.name}(): {exc}"
                ) from exc

            if fname in ANGLE_OUTPUT_FUNCTIONS:
                result = angle_from_radians(result, self.angle_mode)

            return result

        try:
            return info.function(
                *args,
                **kwargs,
            )
        except TypeError as exc:
            raise TypeError(
                f"{info.domain}.{info.name}(): {exc}"
            ) from exc


class RequestParser:
    """Parses the claudemath command syntax."""

    PREFIX = "solve"

    def __init__(
        self,
        expression_evaluator: ExpressionEvaluator,
    ) -> None:
        self.expression_evaluator = expression_evaluator

    def parse(self, request: str) -> Any:
        request = request.strip()

        if not request:
            raise ValueError("Empty request.")

        if not request.lower().startswith(self.PREFIX):
            raise ValueError(
                "Mathematical requests must start with 'solve'."
            )

        remainder = request[len(self.PREFIX):].strip()

        if not remainder:
            raise ValueError(
                "Nothing to solve. Example: solve add(1, 2)"
            )

        # The evaluator already supports full expressions (operators mixed
        # with function calls, e.g. "1 + 1 * sin(30)"), not just a bare
        # "name(...)" call, so no extra shape-check is needed here — any
        # syntax error will surface from evaluate() itself.
        return self.expression_evaluator.evaluate(
            remainder
        )


class ClaudeMathCLI:
    """Interactive claudemath terminal."""

    def __init__(self) -> None:
        self.history: list[str] = []

        self.engine = FunctionEngine()
        self.expression_evaluator = ExpressionEvaluator(
            self.engine
        )
        self.parser = RequestParser(
            self.expression_evaluator
        )

    def evaluate_request(self, request: str) -> None:
        try:
            result = self.parser.parse(request)

            print()
            print(
                color("◆ ", Colors.GREEN)
                + color(
                    self._format_result(result),
                    Colors.WHITE,
                )
            )
            print()

        except KeyboardInterrupt:
            print()
            print(
                color(
                    "Operation cancelled.",
                    Colors.YELLOW,
                )
            )
            print()

        except Exception as exc:
            print()
            print(
                color("✗ ", Colors.RED)
                + color(
                    str(exc),
                    Colors.RED,
                )
            )
            print()

    @staticmethod
    def _format_result(result: Any) -> str:
        if isinstance(result, float):
            if math.isfinite(result):
                if result.is_integer():
                    return str(int(result))

                return f"{result:.15g}"

        return str(result)

    def print_functions(self) -> None:
        functions = self.engine.all_functions()

        print()
        print(
            color(
                f"Available functions ({len(functions)})",
                Colors.BOLD + Colors.CYAN,
            )
        )
        print()

        if not functions:
            print(
                color(
                    "No functions discovered.",
                    Colors.YELLOW,
                )
            )
            print()
            return

        current_domain = None

        for info in functions:
            if info.domain != current_domain:
                current_domain = info.domain

                print(
                    color(
                        f"[{current_domain}]",
                        Colors.MAGENTA,
                    )
                )

            print(
                f"  {color(info.name, Colors.WHITE)}"
            )

        print()

    def print_history(self) -> None:
        print()

        if not self.history:
            print(
                color(
                    "No commands in this session.",
                    Colors.GRAY,
                )
            )
            print()
            return

        print(
            color(
                "Session history",
                Colors.BOLD + Colors.CYAN,
            )
        )
        print()

        for index, request in enumerate(
            self.history,
            1,
        ):
            print(
                f"  {color(f'{index:03}', Colors.GRAY)}"
                f"  {color(request, Colors.WHITE)}"
            )

        print()

    def _handle_mode_command(self, args: list[str]) -> None:
        if not args:
            print()
            print(
                color(
                    f"Current angle mode: {self.expression_evaluator.angle_mode}",
                    Colors.CYAN,
                )
            )
            print(
                color(
                    "Usage: mode <degree|radian|gradian>",
                    Colors.GRAY,
                )
            )
            print()
            return

        try:
            new_mode = self.expression_evaluator.set_angle_mode(args[0])
            save_angle_mode(new_mode)
        except ValueError as exc:
            print()
            print(color(f"✗ {exc}", Colors.RED))
            print()
            return

        print()
        print(
            color(f"◆ Angle mode set to {new_mode}.", Colors.GREEN)
        )
        print(
            color(
                "sin(), cos(), tan(), cot(), sec(), csc(), and their "
                f"inverses now read/return {new_mode}s.",
                Colors.GRAY,
            )
        )
        print()

    def handle_command(self, command: str) -> bool:
        command_name = command.strip().lower()

        parts = command.strip().split()
        first_word = parts[0].lower() if parts else ""

        if first_word in {"/mode", "mode"}:
            self._handle_mode_command(parts[1:])
            return True

        if command_name in {
            "/exit",
            "/quit",
            "/q",
        }:
            return False

        if command_name in {
            "/help",
            "/h",
            "/?",
        }:
            print_help()
            return True

        if command_name == "/clear":
            clear_screen()
            print_banner()
            return True

        if command_name == "/domains":
            print_domains()
            return True

        if command_name == "/functions":
            self.print_functions()
            return True

        if command_name == "/history":
            self.print_history()
            return True

        if command_name == "/version":
            print()
            print(
                color(
                    f"claudemath {VERSION}",
                    Colors.CYAN,
                )
            )
            print()
            return True

        print()
        print(
            color(
                f"Unknown command: {command}",
                Colors.RED,
            )
        )
        print(
            color(
                "Type /help to see available commands.",
                Colors.GRAY,
            )
        )
        print()

        return True

    def run(self) -> int:
        print_banner()

        while True:
            try:
                request = input(
                    color(
                        "claudemath",
                        Colors.CYAN,
                    )
                    + color(
                        " > ",
                        Colors.WHITE,
                    )
                ).strip()

            except KeyboardInterrupt:
                print()
                print(
                    color(
                        "Use /exit to quit.",
                        Colors.GRAY,
                    )
                )
                print()
                continue

            except EOFError:
                print()
                return 0

            if not request:
                continue

            first_word = request.split(maxsplit=1)[0].lower()

            if request.startswith("/") or first_word == "mode":
                if not self.handle_command(request):
                    print()
                    print(
                        color(
                            "Goodbye.",
                            Colors.CYAN,
                        )
                    )
                    print()
                    return 0

                continue

            self.history.append(request)
            self.evaluate_request(request)


def main() -> int:
    argv = sys.argv[1:]

    if argv and argv[0].lower() == "mode":
        if len(argv) == 1:
            print(f"Current angle mode: {load_saved_angle_mode()}")
            print("Usage: claudemath mode <degree|radian|gradian>")
            return 0

        try:
            new_mode = normalize_angle_mode(argv[1])
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        save_angle_mode(new_mode)
        print(f"Angle mode set to {new_mode}.")
        print(
            "This is now the default for new claudemath sessions. "
            f"Use '/mode {new_mode}' to also switch it inside a running session."
        )
        return 0

    cli = ClaudeMathCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
