from fraudfl.config import BankProfile, SimulationConfig
from fraudfl.data.simulate import generate
from fraudfl.graph import build_edge_index


def _tx():
    cfg = SimulationConfig(n_days=20, n_terminals=200, banks=[
        BankProfile("a", (50, 50), 20, (5, 60), (1, 3), target_n_tx=2000),
        BankProfile("b", (50, 50), 20, (5, 60), (1, 3), target_n_tx=2000),
    ])
    return generate(cfg)[2]


def test_edges_point_from_past_to_present():
    tx = _tx()
    ei = build_edge_index(tx, cross_client=True).numpy()
    t = tx["TX_DATETIME"].to_numpy()
    assert (t[ei[0]] <= t[ei[1]]).all()


def test_local_graph_has_no_cross_client_edges():
    tx = _tx()
    c = tx["CLIENT_ID"].to_numpy()
    local = build_edge_index(tx, cross_client=False).numpy()
    central = build_edge_index(tx, cross_client=True).numpy()
    assert (c[local[0]] == c[local[1]]).all()
    assert (c[central[0]] != c[central[1]]).any()   # terminal chung → centralized có cạnh xuyên ngân hàng


def test_edges_link_same_customer_or_terminal():
    tx = _tx()
    ei = build_edge_index(tx, cross_client=True).numpy()
    same_cust = tx["CUSTOMER_ID"].to_numpy()[ei[0]] == tx["CUSTOMER_ID"].to_numpy()[ei[1]]
    same_term = tx["TERMINAL_ID"].to_numpy()[ei[0]] == tx["TERMINAL_ID"].to_numpy()[ei[1]]
    assert (same_cust | same_term).all()
