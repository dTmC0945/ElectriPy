from constants import *

# LOGARITHM CALCULATIONS -----------------------------------------------------------------------------------------------

def ln(x):
    """ Calculates the natural logarithm of the entered value x. Calculates numerically.

    :param x: Entered value
    :return: the natural logarithm of the number
    """
    n = 1000.0  # set precision. Higher better but will impact performance.
    return n * ((x ** (1 / n)) - 1)  # numerically calculates the natural logarithms of x.


def log10(x):
    """ Calculates the common logarithm of the entered value x. Calculates numerically.

        :param x: Entered value
        :return: the common logarithm of the number
        """
    return ln(x) / ln(10)  # numerically calculates the common logarithms of x. Dependent on ln(x) function.


def log2(x):
    """ Calculates the binary logarithm of the entered value x. Calculates numerically.

        :param x: Entered value
        :return: the binary logarithm of the number
        """
    return ln(x) / ln(2)  # numerically calculates the binary logarithms of x. Dependent on ln(x) function.


def logb(base, x):
    """ Calculates the logarithm base b of the entered value x. Calculates numerically.

        :param base: base of the logarithm value
        :param x: Entered value
        :return: the logarithm of the number
        """
    return ln(x) / ln(base)  # numerically calculates the logarithm base b of x. Dependent on ln(x) function.


def infiniteSum(x):
    n, res = 0, x(0)
    while True:
        term = sum(x(k) for k in range(2 ** n, 2 ** (n + 1)))
        if (res + term) - res == 0:
            break
        n, res = n + 1, res + term
    return res


def factorial(n):
    fact = 1
    for num in range(2, n + 1):
        fact *= num
    return fact


# Approximating Trigonometric identities
def sin(theta):
    """ Generates the sin value of the entered value
    :param theta: the angle
    :return: returns the sine of the entered value"""
    return infiniteSum(lambda x: pow(-1, x) * pow(theta, 1 + 2 * x) / factorial(1 + 2 * x))


def cos(theta):
    return infiniteSum(lambda x: pow(-1, x) * pow(theta, 2 * x) / factorial(2 * x))


def tan(theta):
    return sin(theta) / cos(theta)


def cot(theta):
    return cos(theta) / sin(theta)


def sec(theta):
    return 1 / cos(theta)


def cosec(theta):
    return 1 / sin(theta)


def root(num, root):
    global result

    n_dec = 10
    nat_num = 1
    while nat_num ** root <= num:
        result = nat_num
        nat_num += 1
    for d in range(1, n_dec + 1):
        increment = 10 ** -d
        count = 1
        before = result
        while (before + increment * count) ** root <= num:
            result = before + increment * count
            count += 1
    return round(result, n_dec)


def arsinh(x):
    return ln(x + root(x ** 2 + 1, 2))


def arcosh(x):
    return ln(x + root(x ** 2 - 1, 2))


def artanh(x):
    # domain of this function is (-1, 1)
    # uses ln() function defined in this file
    return 0.5 * ln((1 + x) / (1 - x))


def arcoth(x):
    # domain of this function is (-inf, -1) and (1, +inf)
    # uses ln() function defined in this file
    return 0.5 * ln((x + 1) / (x - 1))


def arsech(x):
    # domain of this function is (0, 1]
    # uses ln() and root() functions defined in this file
    return ln(1 / x + root(1 / pow(x, 2) - 1, 2))


def archsch(x):
    return ln(1 / x + root(1 / pow(x, 2) + 1, 2))


def sinh(x):
    return (pow(euler, x) - pow(euler, x)) / 2


def cosh(x):
    return (pow(euler, x) + pow(euler, x)) / 2


def tanh(x):
    return sinh(x) / cosh(x)


def coth(x):
    return cosh(x) / sinh(x)


def gammaFunction(x):
    return factorial(x - 1)


def binomialCoefficient(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))


def riemannZeta(s):
    return infiniteSum(lambda n: 1 / pow(n, s))


def integrate(f, a, b, dx=0.1):
    """ Does numerical integration. Fairly accurate. Wouldn't trust more than the 7th decimal

    :param f: Function (i.e., lambda x: x)
    :param a: lower bound
    :param b: upper bound
    :param dx: the size of the rectangles used to calculate the area under the curve
    :return: the integral value.
    """
    i = a
    s = 0
    while i <= b:
        s += f(i) * dx
        i += dx
    return s


def asin(x):
    """ Produces the angle value in radians based on the entered value ranging between (-1, +1)

    :param x: The entered value randing between [-1, +1]
    :return: Produced the angle value that would have generated the x value [-pi/2, +pi/2]
    """
    return infiniteSum(lambda n: 1 / pow(2, 2 * n) * binomialCoefficient(2 * n, n) * pow(x, 2 * n + 1) / (2 * n + 1))


def erf(z):
    """ Produces the error function value of the entered value.

    :param z: The upper range value. (the lower value is set to 0)
    :return: returns the erf(x) value within the range of [-1, 1].
    """
    return 2 / root(pi, 2) * integrate(lambda t: pow(euler, -pow(t, 2)), 0, z, 0.001)


def erfc(z):
    """ Returns the complementary error function (erf(z)).

    :param z: The entered value
    :return: The complementary error function. Range between [-1, +1].
    """
    return 1 - erf(z)


def phasor(A, theta, *args):
    """Calculate the phasor value. When no argument is set Radian is taken as default.

    :param A: takes magnitude
    :param theta: takes angle (can be degree or radian based on argument)
    :param args: "Degree", "Radian"
    :return: complex value"""

    if not args:
        return complex(A * cos(theta), A * sin(theta))
    elif args[0] == 'Degree':
        return complex(A * cos(theta * pi / 180), A * sin(theta * pi / 180))
    elif args[0] == 'Radian':
        return complex(A * cos(theta), A * sin(theta))
    else:
        raise ValueError("The arguments can either be Degree or Radian.")


def ramanujanPi():
    """Calculates the value of pi using the Ramanujan approximation using infinite sum. Takes no value.
    You may ask why ? I say why not ? Is it useful... nope"""
    return 1 / (infiniteSum(
        lambda k: factorial(4 * k) * (1103 + 26390 * k) / (pow((factorial(k)), 4) * pow(396, 4 * k))) * 2 * root(2,
                                                                                                                 2) / 9801)


def exp(x):
    """Produces the exponential value of the entered value x

    :param x: takes a value"""
    return pow(euler, x)


def normalDist(x, mu, sigma):
    return 1 / (sigma * root(2 * pi, 2)) * exp(-1 / 2 * pow((x - mu) / sigma, 2))
