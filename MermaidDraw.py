from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod

router = AgentRouterGraph()
router.graph.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
    output_file_path="workflow.png"
)