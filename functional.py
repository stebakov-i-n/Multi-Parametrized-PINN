from torch import autograd


def calc_grad(outputs, x):
    dv1 = autograd.grad(outputs[..., 0].sum(), x, create_graph=True)[0]
    dv2 = autograd.grad(outputs[..., 1].sum(), x, create_graph=True)[0]
    dp = autograd.grad(outputs[..., 2].sum(), x, create_graph=True)[0]
    d2v1 = autograd.grad(dv1.sum(), x, create_graph=True)[0]
    d2v2 = autograd.grad(dv2.sum(), x, create_graph=True)[0]
    return dv1, dv2, d2v1, d2v2, dp


def calc_res(outputs, dv1, dv2, d2v1, d2v2, dp, mu, rho):
    res1 = (outputs[..., 0] * dv1[..., 0] + outputs[..., 1] * dv1[..., 1]) - mu * (d2v1[..., 0] + d2v1[..., 1]) / rho + dp[..., 0] / rho
    res2 = (outputs[..., 0] * dv2[..., 0] + outputs[..., 1] * dv2[..., 1]) - mu * (d2v2[..., 0] + d2v2[..., 1]) / rho + dp[..., 1] / rho
    return [res1, res2]


def int_loss(dv1, dv2, mu):
    """
        Strain rate tensor [Txi] and the shear rate intensity [Eta^2] distribution in the flow domain
    """

    # Txi
    xi11 = dv1[0]
    xi12 = 0.5 * (0 + 0)
    xi13 = 0.5 * (dv1[..., 2] + dv2[..., 0])

    xi22 = 0
    xi23 = 0.5 * (0 + dv2[..., 1])

    xi33 = dv2[..., 2]

    xi0 = (xi11 + xi22 + xi33) / 3

    xi11 = xi11 - xi0
    xi22 = xi22 - xi0
    xi33 = xi33 - xi0

    # Eta^2
    eta_square = (2 * (xi11 * xi11 + xi22 * xi22 + xi33 * xi33 +
                  2 * (xi12 * xi12 + xi13 * xi13 + xi23 * xi23)))

    return (0.5 * mu * eta_square).mean()


def calc_div(dv1, dv2):
    return dv1[..., 0] + dv2[..., 1]


def mse_zero_loss(f):
    return (f ** 2).mean()


def zero_loss(outputs, n=2):
    loss = 0
    for i in range(n):
        loss += mse_zero_loss(outputs[i])
    loss = loss / n
    return loss