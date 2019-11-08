from simulation_runner import *
from simulation import *
from decision_maker import *
from network import Network


def main():
    np.random.seed(1)
    simulation = Simulation()
    configure_simulation(simulation)
    runner = SimulationRunner(simulation)

    for i in range(0, 100):
        # raw_input("press any key to continue...")
        runner.next_cycle()
        print simulation.to_json()


def configure_simulation(simulation):
    configure_agents(simulation)
    configure_network(simulation)


def configure_agents(simulation):
    num_hc = 2
    num_ws = 2
    num_ds = 2
    num_mn = 2

    for i in range(0, num_mn):
        mn = Manufacturer()
        mn.id = i
        mn.num_of_active_lines = 2
        mn.line_capacity = 60
        mn.up_to_level = 240
        mn.decisionMaker = SimpleMNDecisionMaker(mn)
        simulation.manufacturers.append(mn)

    for i in range(0, num_ds):
        ds = Distributor()
        ds.id = i + num_mn
        ds.decisionMaker = SimpleDSDecisionMaker(ds)
        simulation.distributors.append(ds)

    for i in range(0, num_ws):
        ws = Wholesaler()
        ws.id = i + num_mn + num_ds
        ws.decisionMaker = SimpleWSDecisionMaker(ws)
        simulation.distributors.append(ws)


    for i in range(0, num_hc):
        hc = HealthCenter()
        hc.id = i + num_mn + num_ds + num_ws
        hc.up_to_level=240
        hc.decisionMaker = SimpleHCDecisionMaker(hc)
        simulation.health_centers.append(hc)


def configure_network(simulation):
    simulation.healthCenters[0].upstream_nodes.extend(simulation.distributors)
    simulation.healthCenters[0].upstream_nodes.append(simulation.manufacturers[0])
    simulation.healthCenters[1].upstream_nodes.append(simulation.distributors[1])
    simulation.distributors[0].upstream_nodes.append(simulation.manufacturers[1])
    simulation.distributors[1].upstream_nodes.extend(simulation.manufacturers)
    simulation.distributors[0].downstream_nodes.append(simulation.health_centers[0])
    simulation.distributors[1].downstream_nodes.extend(simulation.health_centers)
    simulation.manufacturers[0].downstream_nodes.append(simulation.distributors[1])
    simulation.manufacturers[0].downstream_nodes.append(simulation.health_centers[0])
    simulation.manufacturers[1].downstream_nodes.extend(simulation.distributors)



    net = Network(6)
    simulation.network = net
    net.connectivity[0, 3] = 2
    net.connectivity[0, 4] = 2
    net.connectivity[1, 2] = 2
    net.connectivity[1, 3] = 2
    net.connectivity[2, 4] = 2
    net.connectivity[3, 4] = 2
    net.connectivity[3, 5] = 2

    info_net = Network(6)
    simulation.info_network = info_net
    for i in range(0, 6):
        for j in range(0, 6):
            info_net.connectivity[i, j] = 0

if __name__ == "__main__":
    main()
