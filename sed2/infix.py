import numpy as np
import re


def infix_to_postfix(infix_expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    right_associative = {'^'}
    stack = []
    postfix = []

    # Tokenizing the infix expression
    tokens = re.findall(r"[\w.]+|[^ \w]", infix_expression)

    for token in tokens:
        if re.match(r"^[\d.]+$", token):  # If the token is a number (integer or float)
            postfix.append(token)
        elif token.startswith('np'):  # If the token is a numpy array
            postfix.append(token)
        elif token == '(':
            stack.append('(')
        elif token == ')':
            while stack and stack[-1] != '(':
                postfix.append(stack.pop())
            if stack:
                stack.pop()
        else:  # If the token is an operator
            while (stack and stack[-1] != '(' and
                   ((token not in right_associative and precedence.get(token, 0) <= precedence.get(stack[-1], 0)) or
                    (token in right_associative and precedence.get(token, 0) < precedence.get(stack[-1], 0)))):
                postfix.append(stack.pop())
            stack.append(token)

    while stack:
        postfix.append(stack.pop())

    postfix_expression = " ".join(postfix)
    print(f"Postfix Expression: {postfix_expression}")  # Debugging print statement
    return postfix_expression


def evaluate_postfix(postfix_expression):
    stack = []
    tokens = postfix_expression.split()

    for token in tokens:
        if re.match(r"^[\d.]+$", token):  # If the token is a number (integer or float)
            stack.append(float(token))
        elif token.startswith('np'):  # If the token is a numpy array
            stack.append(eval(token))
        else:  # If the token is an operator
            val1 = stack.pop()
            val2 = stack.pop()
            switcher = {
                '+': val2 + val1,
                '-': val2 - val1,
                '*': val2 * val1,
                '/': val2 / val1,
                '^': val2 ** val1
            }
            stack.append(switcher.get(token))

    print(f"Final Stack: {stack}")  # Debugging print statement
    return stack[0]


def evaluate_infix(infix_expression):
    postfix_expression = infix_to_postfix(infix_expression)
    result = evaluate_postfix(postfix_expression)
    return result
