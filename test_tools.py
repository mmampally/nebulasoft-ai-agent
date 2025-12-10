from tools import DocSearchTool, PricingCalculatorTool, TicketEscalationTool


def test_doc_search():
    print("=" * 60)
    print("✅ Testing DocSearchTool (NebulaSoft KB)")
    print("=" * 60)

    tool = DocSearchTool()

    query = "error 500"
    print(f"Query: {query}\n")

    result = tool._run(query)
    print("Result:")
    print(result)
    print()


def test_pricing():
    print("=" * 60)
    print("✅ Testing PricingCalculatorTool")
    print("=" * 60)

    tool = PricingCalculatorTool()

    result = tool._run(number_of_users=10, plan_type="pro")
    print(result)
    print()


def test_ticket():
    print("=" * 60)
    print("✅ Testing TicketEscalationTool")
    print("=" * 60)

    tool = TicketEscalationTool()

    result = tool._run(
        summary="User cannot connect to database after upgrade",
        severity_level="high"
    )
    print(result)
    print()


if __name__ == "__main__":
    print("\n🔍 RUNNING ALL TOOL TESTS (LOCAL MODE)\n")

    test_doc_search()
    test_pricing()
    test_ticket()

    print("✅ ALL TESTS COMPLETED\n")
