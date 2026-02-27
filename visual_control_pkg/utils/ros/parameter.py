from typing import Any

from rcl_interfaces.msg import ParameterType, ParameterValue


def python_to_param_value(value: Any) -> ParameterValue:
    """Convert from a Python value to a ParameterValue.

    Supported types are only `bool`, `int`, `float`, `str` and **homogeneous** lists of those types.
    **No** other types are supported and **will** raise a `KeyError` exception.

    Args:
        value: Python variable to convert whose type is one of the types listed above.

    Returns:
        Corresponding ParameterValue to the given input whose type and value fields are set
        appropiately.
    """
    key = type(value)
    if isinstance(value, list):
        key = list[type(value[0])]
    param_type = _PYTHON_TYPE_TO_PARAMETER_TYPE[key]
    param_attr = _PYTHON_TYPE_TO_PARAMETER_ATTRIBUTE[key]
    return ParameterValue(**{"type": param_type, param_attr: value})


_PYTHON_TYPE_TO_PARAMETER_TYPE = {
    bool: ParameterType.PARAMETER_BOOL,
    int: ParameterType.PARAMETER_INTEGER,
    float: ParameterType.PARAMETER_DOUBLE,
    str: ParameterType.PARAMETER_STRING,
    list[bool]: ParameterType.PARAMETER_BOOL_ARRAY,
    list[int]: ParameterType.PARAMETER_INTEGER_ARRAY,
    list[float]: ParameterType.PARAMETER_DOUBLE_ARRAY,
    list[str]: ParameterType.PARAMETER_STRING_ARRAY,
}


_PYTHON_TYPE_TO_PARAMETER_ATTRIBUTE = {
    bool: "bool_value",
    int: "integer_value",
    float: "double_value",
    str: "string_value",
    list[bool]: "bool_array_value",
    list[int]: "integer_array_value",
    list[float]: "double_array_value",
    list[str]: "string_array_value",
}
