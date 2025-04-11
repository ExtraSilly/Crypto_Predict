from django import template

register = template.Library()

@register.filter
def subtract(value, arg):
    """Subtract the arg from the value."""
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def divide(value, arg):
    """Divide the value by the arg."""
    try:
        return float(value) / float(arg)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0

@register.filter
def mul(value, arg):
    """Multiply the value by the arg."""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return ''

@register.filter
def calculate_change(current, predicted):
    """Calculate percentage change between current and predicted values."""
    try:
        current = float(current)
        predicted = float(predicted)
        if current == 0:
            return 0
        change = ((predicted - current) / current) * 100
        return round(change, 2)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0
