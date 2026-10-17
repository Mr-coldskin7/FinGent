"""
Tests for LLM/router.py — Router (pure logic, no external deps).
"""



class TestRouterRoute:
    """Test Router.route() intent → agent mapping."""

    def _make_preprocess(self, intent_type, status="ready"):
        return {
            "status": status,
            "intent": {"type": intent_type},
            "entities": {"symbols": ["AAPL"]},
        }

    def test_technical_analysis(self, router):
        result = router.route(self._make_preprocess("TECHNICAL_ANALYSIS"))
        assert result == "technical-nerd"

    def test_company_info(self, router):
        result = router.route(self._make_preprocess("COMPANY_INFO"))
        assert result == "morefit"

    def test_report_analysis(self, router):
        result = router.route(self._make_preprocess("REPORT_ANALYSIS"))
        assert result == "morefit"

    def test_news_sentiment(self, router):
        result = router.route(self._make_preprocess("NEWS_SENTIMENT"))
        assert result == "sentiment-analyzer"

    def test_sentiment_check(self, router):
        result = router.route(self._make_preprocess("SENTIMENT_CHECK"))
        assert result == "sentiment-analyzer"

    def test_risk_management(self, router):
        result = router.route(self._make_preprocess("RISK_MANAGEMENT"))
        assert result == "risk-manager"

    def test_portfolio_risk(self, router):
        result = router.route(self._make_preprocess("PORTFOLIO_RISK"))
        assert result == "risk-manager"

    def test_market_data_goes_to_all(self, router):
        result = router.route(self._make_preprocess("MARKET_DATA"))
        assert result == "ALL"

    def test_suggestions_goes_to_all(self, router):
        result = router.route(self._make_preprocess("SUGGESTIONS"))
        assert result == "ALL"

    def test_unknown_intent(self, router):
        result = router.route(self._make_preprocess("FOOBAR"))
        assert result == "unknown"

    def test_clarification_needed(self, router):
        preprocess = {
            "status": "clarification_needed",
            "intent": {"type": "MARKET_DATA"},
            "entities": {},
        }
        result = router.route(preprocess)
        assert result == "clarify_node"

    def test_clarification_overrides_intent(self, router):
        """Even if intent is valid, clarification_needed should return clarify_node."""
        preprocess = {
            "status": "clarification_needed",
            "intent": {"type": "TECHNICAL_ANALYSIS"},
            "entities": {},
        }
        result = router.route(preprocess)
        assert result == "clarify_node"


class TestRouteTable:
    """Verify the ROUTES table has all expected entries."""

    def test_all_routes_present(self, router):
        expected_keys = {
            "MARKET_DATA",
            "COMPANY_INFO",
            "REPORT_ANALYSIS",
            "NEWS_SENTIMENT",
            "TECHNICAL_ANALYSIS",
            "RISK_MANAGEMENT",
            "PORTFOLIO_RISK",
            "SENTIMENT_CHECK",
            "SUGGESTIONS",
        }
        assert set(router.ROUTES.keys()) == expected_keys

    def test_all_route_values_are_valid_agents(self, router):
        """ROUTES 的值必须是注册表里的 agent 或者 ALL。"""
        from LLM.registry import all_agents

        valid_agents = set(all_agents().keys()) | {"ALL"}
        for agent in router.ROUTES.values():
            assert agent in valid_agents, f"Unknown agent in ROUTES: {agent}"
