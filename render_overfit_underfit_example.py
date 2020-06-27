from render_preamble import *

# plt.figure(num=None, figsize=(8, 6), dpi=80)

n = 25
xs = np.asarray( [ x / n for x in range(math.floor(2 * math.pi * n)) ] )
ys = [ math.sin(x) + 0.15 * np.random.normal() for x in xs ]

df = pd.DataFrame({
    'xs': xs,
    'ys': ys,
})

plt.tight_layout()
sn.scatterplot('xs', 'ys', data=df)
plt.xlabel("")
plt.ylabel("")

################################################################################

xs = np.reshape(xs, (-1,1))

# Bake up some terrible classifiers
from sklearn.tree import DecisionTreeRegressor
overfit = DecisionTreeRegressor().fit(xs, ys)
overfit_ys = overfit.predict(xs)

from sklearn.linear_model import LinearRegression
underfit = LinearRegression().fit(xs[0:1,:], ys[0:1])
underfit_ys = underfit.predict(xs)

expected_ys = [ math.sin(x) for x in xs ]

df = pd.DataFrame({
    'xs': xs[:,0],
    'underfitted': underfit_ys,
    'overfitted': overfit_ys,
    'expected': expected_ys,
})
df = pd.melt(df, ['xs'])

plt.tight_layout()
sn.lineplot('xs', 'value', 'variable', data=df)
plt.xlabel("")
plt.ylabel("")
plt.savefig('renders/background/overfit_underfit.png')

