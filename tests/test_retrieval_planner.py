from c3ae.retrieval.planner import plan_memory_query


def test_planner_routes_current_state_queries():
    plan = plan_memory_query("What drink does the user currently prefer?")

    assert plan.route == "current_state"
    assert plan.answer_type == "preference"
    assert "active_memory_preference" in plan.strategies


def test_planner_routes_table_queries():
    plan = plan_memory_query("In the schedule table, which shift is Sarah assigned?")

    assert plan.route == "table_lookup"
    assert "markdown_table_reader" in plan.strategies


def test_planner_routes_temporal_math_queries():
    plan = plan_memory_query("How many days passed between June 1 and June 9?")

    assert plan.route == "temporal_math"
    assert "numeric_solver" in plan.strategies


def test_planner_routes_cross_session_queries():
    plan = plan_memory_query("What did I mention before in our previous chat?")

    assert "cross_session" in plan.intents
    assert "episode_retrieval" in plan.strategies

