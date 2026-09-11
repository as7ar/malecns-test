from dotenv import load_dotenv
from neuprint import Client, fetch_adjacencies
import os
import malecns

load_dotenv()

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=os.getenv("TOKEN"),
)

neuron_info, connections = fetch_adjacencies(
    "DNge104",
    None,
    omit_rois=True,
)

neuron_ids = set()

for _, row in connections.iterrows():
    neuron_ids.add(int(row["bodyId_pre"]))
    neuron_ids.add(int(row["bodyId_post"]))

neuron_ids = list(neuron_ids)

id_to_index = {
    neuron_id: index
    for index, neuron_id in enumerate(neuron_ids)
}

rust_connections = []

for _, row in connections.iterrows():
    from_id = int(row["bodyId_pre"])
    to_id = int(row["bodyId_post"])
    weight = float(row["weight"])

    rust_connections.append((
        id_to_index[from_id],
        id_to_index[to_id],
        weight,
    ))

network = malecns.PyNetwork(len(neuron_ids))

network.add_connections(rust_connections)

print("Neurons:", len(neuron_ids))
print("Connections:", len(rust_connections))

network.set_input(0, 1.0)

for i in range(10):
    network.step()

    print(
        f"Step {i}:",
        network.get_potential(0),
    )
