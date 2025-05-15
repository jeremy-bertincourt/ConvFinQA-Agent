from pydantic import BaseModel, ValidationError, Field
from langchain.tools import Tool
from typing import Any, Dict

class SubtractToolBuilder:
    """
    Provides a tool to subtract two numbers, with input validation.
    """
    class SubtractArgs(BaseModel):
        arg1: float = Field(..., description="new value in substraction")
        arg2: float = Field(..., description="old value in substraction")

    @staticmethod
    def build_tool() -> Tool:
        def subtract(raw: str) -> str:
            """
            Subtracts the second number from the first.
            Input format: 'arg1, arg2' (e.g., '60.94, 25.14')
            """
            try:
                # 1) Parse the raw string into two floats
                parts = [p.strip() for p in raw.replace("'", "").split(",")]
                if len(parts) != 2:
                    raise ValueError("Expected exactly two comma-separated values")
                arg1_str, arg2_str = parts
                # 2) Validate & coerce using Pydantic
                args = SubtractToolBuilder.SubtractArgs(
                    arg1=float(arg1_str),
                    arg2=float(arg2_str),
                )
            except ValueError as e:
                return (
                    "Error: could not parse inputs. "
                    "Provide exactly two numbers, e.g. '60.94, 25.14'."
                )
            except ValidationError as e:
                # Pydantic catch: wrong types, missing values, etc.
                return f"Error: invalid arguments for subtract(): {e}"

            # 3) Perform the subtraction
            result = args.arg1 - args.arg2
            return str(result)

        return Tool(
            name="subtract",
            func=subtract,
            description=(
                "Use this tool to subtract two numbers."
                "Input must be a string: 'arg1, arg2', e.g. '60.94, 25.14'. "
                "This tool validates that both are numeric."
            )
        )


class AddToolBuilder:
    class AddArgs(BaseModel):
        arg1: float = Field(..., description="First value")
        arg2: float = Field(..., description="Second value")
    @staticmethod
    def build_tool() -> Tool:
        def add(raw: str) -> str:
            """
            Adds the second number to the first.
            Input format: 'arg1, arg2' (e.g., '60.94, 25.14')
            """
            try:
                parts = [p.strip() for p in raw.replace("'", "").split(",")]
                if len(parts) != 2:
                    raise ValueError("Expected exactly two comma-separated values")
                args = AddToolBuilder.AddArgs(
                    arg1=float(parts[0]),
                    arg2=float(parts[1]),
                )
            except ValueError:
                return (
                    "Error: could not parse inputs. "
                    "Provide exactly two numbers, e.g. '5.0, 3.2'."
                )
            except ValidationError as e:
                return f"Error: invalid arguments for add(): {e}"
            return str(args.arg1 + args.arg2)

        return Tool.from_function(
            add,
            name="add",
            description=(
                "Use this tool to add two numbers."
                "Add(arg1, arg2) → float. Input must be 'arg1, arg2', "
                "e.g. '5.0, 3.2'."
            ),
        )


class MultiplyToolBuilder:
    class MultiplyArgs(BaseModel):
        arg1: float = Field(..., description="First factor")
        arg2: float = Field(..., description="Second factor")
    @staticmethod
    def build_tool() -> Tool:
        def multiply(raw: str) -> str:
            """
            Multiplies the second number by the first.
            Input format: 'arg1, arg2' (e.g., '60.94, 25.14')
            """
            try:
                parts = [p.strip() for p in raw.replace("'", "").split(",")]
                if len(parts) != 2:
                    raise ValueError("Expected exactly two comma-separated values")
                args = MultiplyToolBuilder.MultiplyArgs(
                    arg1=float(parts[0]),
                    arg2=float(parts[1]),
                )
            except ValueError:
                return (
                    "Error: could not parse inputs. "
                    "Provide exactly two numbers, e.g. '4, 2.5'."
                )
            except ValidationError as e:
                return f"Error: invalid arguments for multiply(): {e}"
            return str(args.arg1 * args.arg2)

        return Tool.from_function(
            multiply,
            name="multiply",
            description=(
                "Use this tool to multiply two numbers."
                "Multiply(arg1, arg2) → float. Input must be 'arg1, arg2', "
                "e.g. '4, 2.5'."
            ),
        )


class DivideToolBuilder:
    class DivideArgs(BaseModel):
        arg1: float = Field(..., description="new value (numerator)")
        arg2: float = Field(..., description="old value (denominator)")
    @staticmethod
    def build_tool() -> Tool:
        def divide(raw: str) -> str:
            """
            Divides the second number from the first.
            Input format: 'arg1, arg2' (e.g., '60.94, 25.14')
            """
            try:
                parts = [p.strip() for p in raw.replace("'", "").split(",")]
                if len(parts) != 2:
                    raise ValueError("Expected exactly two comma-separated values")
                args = DivideToolBuilder.DivideArgs(
                    arg1=float(parts[0]),
                    arg2=float(parts[1]),
                )
            except ValueError:
                return (
                    "Error: could not parse inputs. "
                    "Provide exactly two numbers, e.g. '10.0, 2.0'."
                )
            except ValidationError as e:
                return f"Error: invalid arguments for divide(): {e}"

            if args.arg2 == 0:
                return "Error: division by zero is not allowed"
            return str(args.arg1 / args.arg2)

        return Tool.from_function(
            divide,
            name="divide",
            description=(
                "Use this tool to divide two numbers."
                "Divide(arg1, arg2) → float. Input must be 'arg1, arg2', "
                "e.g. '10.0, 2.0'. Division by zero is checked."
            ),
        )

