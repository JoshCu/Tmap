import pandas as pd
import tmap
from faerun import Faerun
from mhfp.encoder import MHFPEncoder
from rdkit.Chem import AllChem
from rdkit.Chem.Lipinski import NumAromaticRings
from rdkit import Chem
import json

#url = 'https://raw.githubusercontent.com/XinhaoLi74/molds/master/clean_data/ESOL.csv'

# load the data from smiles.txt, one smile per line
with open('smiles.txt', 'r') as f:
    smiles = f.readlines()
    smiles = [s.strip() for s in smiles]

# load the data into a pandas dataframe
df = pd.DataFrame(smiles, columns=['smiles'])

# add a column for the number of AromaticRings
df['AromaticRings'] = 0

print(df.shape)


# The number of permutations used by the MinHashing algorithm
perm = 512

# Initializing the MHFP encoder with 512 permutations
enc = MHFPEncoder(perm)

# Create MHFP fingerprints from SMILES
# The fingerprint vectors have to be of the tm.VectorUint data type
fingerprints = []
for i, f in enumerate(df["smiles"]):
    try:
        mol = Chem.MolFromSmiles(f)
        if mol is None:
            print("Error: ", f)
        df["AromaticRings"][i] = NumAromaticRings(mol)
        fingerprints.append(tmap.VectorUint(enc.encode(f)))
    except:
        print("Error: ", f)

# fingerprints = [tmap.VectorUint(enc.encode(s)) for s in df["smiles"]]

# Initialize the LSH Forest
lf = tmap.LSHForest(perm)

# Add the Fingerprints to the LSH Forest and index
lf.batch_add(fingerprints)
lf.index()

# Get the coordinates
x, y, s, t, _ = tmap.layout_from_lsh_forest(lf)

# Now plot the data
faerun = Faerun(view="front", coords=False)
faerun.add_scatter(
    "Odor_Basic",
    {"x": x,
        "y": y,
     "c": list(df.AromaticRings.values),
        "labels": df["smiles"]},
    point_scale=1,
    colormap=['rainbow'],
    has_legend=True,
    legend_title=['Number of Aromatic Rings'],
    categorical=[False],
    shader='smoothCircle'
)

faerun.add_tree("Odor_Basic_tree", {"from": s, "to": t}, point_helper="Odor_Basic")

# Choose the "smiles" template to display structure on hover
faerun.plot('Odor_Basic', template="smiles", notebook_height=750)
