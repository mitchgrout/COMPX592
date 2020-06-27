from render_preamble import *

n = 25
xs = np.asarray( [ x / n for x in range(math.floor(2 * math.pi * n)) ] )
ys = []
category = []

for x in xs:
    noise = np.random.normal()
    if abs(noise) > 1.0:
        ys.append( math.sin(x) + 0.15 * noise )
        category.append('anomaly')
    else:
        ys.append( math.sin(x) + 0.15 * noise )
        category.append('normal')

df = pd.DataFrame({
    'xs': xs,
    'ys': ys,
    'variable': category,
})

sn.scatterplot('xs', 'ys', 'variable', data=df)
plt.xlabel("")
plt.ylabel("")
plt.savefig("renders/background/anomaly_example.png")

