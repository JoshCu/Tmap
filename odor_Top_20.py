import time
import pandas as pd
import tmap
from faerun import Faerun
from mhfp.encoder import MHFPEncoder
from rdkit.Chem import AllChem
from rdkit.Chem.Lipinski import NumAromaticRings
from rdkit import Chem
import json
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.decomposition import PCA


@dataclass
class molecule:
    smiles: str
    odor: list
    cid: int


# load the data from smiles_odor.json
with open('smiles_odor.json', 'r') as f:
    smiles = json.load(f)


# make all the descriptors lower case and count occurence of each
# get the molecules in the same loop to save time
descriptors = {}
molecules = []
for s in smiles:
    smile = s['_id']
    cid = s['CID']
    odors = []
    for source, values in s['odor'][0].items():
        for value in values:
            for d in value:
                d = d.lower()
                odors.append(d)
                if d in descriptors:
                    descriptors[d] += 1
                else:
                    descriptors[d] = 1
    odors = list(set(odors))
    molecules.append(molecule(smile, odors, cid))
# Blame mongo for this mess

# sort the descriptors by occurence
descriptors = sorted(descriptors.items(), key=lambda x: x[1], reverse=True)

# keep only the top 20 descriptors
# info loss acceptable for visualization clarity
descriptors = descriptors[:20]

# back to a dictionary for easy lookup
descriptors = {d[0]: d[1] for d in descriptors}

# onehot the molecule descriptors
onehot = []
for m in tqdm(molecules):
    molecule_dict = {}
    molecule_dict['smiles'] = m.smiles
    molecule_dict['cid'] = m.cid
    for descriptor in descriptors:
        if descriptor in m.odor:
            molecule_dict[descriptor] = 1
        else:
            molecule_dict[descriptor] = 0
    onehot.append(molecule_dict)

# load the data into a pandas dataframe
df = pd.DataFrame(onehot, columns=['smiles', 'cid'] + list(descriptors.keys()))

pca = PCA(n_components=1)
df['pca'] = pca.fit_transform(df[list(descriptors.keys())])


print(df.shape)
print(df.head())

# print(df.shape)


# The number of permutations used by the MinHashing algorithm
perm = 512

# Initializing the MHFP encoder with 512 permutations
enc = MHFPEncoder(perm)

# Create MHFP fingerprints from SMILES
# The fingerprint vectors have to be of the tm.VectorUint data type
fingerprints = []
for i, f in tqdm(enumerate(df["smiles"])):
    try:
        mol = Chem.MolFromSmiles(f)
        if mol is None:
            print("Error: ", f)
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
labels = [key for key in df.keys() if key != "smiles" and key != "cid" and key != "pca"]

# Now plot the data

# start a timer
start = time.time()

faerun = Faerun(view="front", coords=False)
faerun.add_scatter(
    "Odor_Top_20",
    {"x": x,
        "y": y,
     "c": [list(df.pca.values), [0, 1]],
        "labels": df["smiles"]},
    point_scale=2,
    colormap=['rainbow', 'Set1'],
    has_legend=True,
    legend_title=labels,
    series_title=labels,
    legend_labels=[None, [(0, "No"), (1, "Yes")]],
    categorical=[False, True],
    shader='smoothCircle'
)

faerun.add_tree("Odor_Top_20_tree", {"from": s, "to": t}, point_helper="Odor_Top_20")

# Choose the "smiles" template to display structure on hover
faerun.plot('Odor_Top_20', template="smiles", notebook_height=750)

# print the time taken
print("Time taken for plot: ", time.time() - start)
