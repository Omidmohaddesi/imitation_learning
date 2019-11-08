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
    num_hc = 1
    num_ws = 1
    num_ds = 1
    num_mn = 1

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
    simulation.healthCenters[0].upstream_nodes.extend(simulation.wholesalers)
    simulation.healthCenters[0].upstream_nodes.append(simulation.manufacturers[0])
    simulation.wholesalers[0].upstream_nodes.extend(simulation.distributors)
    simulation.wholesalers[0].upstream_nodes.append(simulation.distributors[0])
    simulation.distributors[0].upstream_nodes.extend(simulation.manufacturers)
    simulation.distributors[0].upstream_nodes.append(simulation.manufacturers[0])
    simulation.manufacturers[0].downstream_nodes.append(simulation.distributors[0])
    simulation.distributors[0].downstream_nodes.append(simulation.wholesalers[0])
    simulation.wholesalers[0].downstream_nodes.append(simulation.health_centers[0])



    net = Network(4)
    simulation.network = net
    net.connectivity[0, 1] = 2
    net.connectivity[1, 2] = 2
    net.connectivity[2, 3] = 2
    net.connectivity[3, 4] = 2


    info_net = Network(4)
    simulation.info_network = info_net
    for i in range(0, 4):
        for j in range(0, 4):
            info_net.connectivity[i, j] = 0

if __name__ == "__main__":
    main()
