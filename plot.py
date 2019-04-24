import numpy as np
import matplotlib.pyplot as plt
import argparse
import re

r = re.compile(r'(?:(sqrt[(](.*)[)])|((.{1})([*]{2})(.{1}))|((.{1})([*]{1})(.{1})))')

def main():

    parser = argparse.ArgumentParser(description='Plotting equations')
    parser.add_argument('-lo','--lower',type=int,default=0,dest='lower_bound',help='Specify lower bound for plotting')
    parser.add_argument('-hi','--higher',type=int,default=100,dest='upper_bound',help='Specify upper bound for plotting')
    parser.add_argument('-p','--points',type=int,default=1000,dest='point_limit',help='Number of points to plot for line')
    parser.add_argument(dest='eq_list',metavar=r'f(x)',type=str,nargs='+')
    args = parser.parse_args()
    return args


args = main()

LOWER_BOUND = args.lower_bound
UPPER_BOUND = args.upper_bound
POINTS = args.point_limit
EQ_LIST = args.eq_list

def parse_equation(equation):
    parsed = r'$' + equation + '$'

    matches = r.finditer(parsed)
    for match in matches:
        groups = [x for x in match.groups() if x != None]

        for group in groups:
            if group.startswith('sqrt'):
                parsed = parsed.replace(groups[0],'sqrt{%s}' % (groups[1]))
                break
            elif group == '**':
                parsed = parsed.replace(groups[0],'sqrt{%s}' % (groups[1]))
                break
            elif group == '*':
                parsed = parsed
                break

    return parsed

assert LOWER_BOUND < UPPER_BOUND , '-> invalid range.'

x0 = np.linspace(LOWER_BOUND,UPPER_BOUND,POINTS)
y0 = np.array([])

legends = []

for equation in EQ_LIST:
    legends.append(parse_equation(equation))
    for x in x0:
        eval_func = equation.replace('x',str(x))
        y0 = np.append(y0,eval(eval_func))
    plt.plot(x0,y0)
    y0 = np.array([])

plt.axis([LOWER_BOUND,UPPER_BOUND,LOWER_BOUND,UPPER_BOUND])
plt.legend(legends)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()
