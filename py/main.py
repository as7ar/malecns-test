import os

from dotenv import load_dotenv
from neuprint import Client, fetch_neurons, fetch_adjacencies


load_dotenv()

token = os.getenv("TOKEN")

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=token,
)

neurons, synapses = fetch_neurons("DNge104")

print(neurons)
print(synapses)

outgoing, neuron_info = fetch_adjacencies("DNge104")

print("Outgoing:")
print(outgoing)

incoming, neuron_info = fetch_adjacencies(None, "DNge104")

print("Incoming:")
print(incoming)
