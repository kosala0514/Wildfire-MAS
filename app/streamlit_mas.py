"""Minimal Streamlit front end for the MAS (merge into your existing UI).

    streamlit run app/streamlit_mas.py

Shows the answer, the plan with per-step notes (IORS rounds, fallbacks), every evidence
item with its query/rows, a map of hotspots, and token/latency cost.
"""
import asyncio

import pandas as pd
import streamlit as st

from wildfire_mas.agents.state import Request
from wildfire_mas.orchestrator import build

st.set_page_config(page_title="Wildfire screening", layout="wide")
st.title("Wildfire screening (multi-agent)")


@st.cache_resource
def orchestrator():
    return build()


with st.sidebar:
    record = st.text_input("Selected record IRI (optional)")
    place = st.text_input("Place name (optional)")
    c1, c2 = st.columns(2)
    lat = c1.number_input("Lat", value=None, format="%.4f")
    lon = c2.number_input("Lon", value=None, format="%.4f")
    role = st.selectbox("Role", ["analyst", "incident_commander", "public"])

q = st.chat_input("Ask about fires, hazards, FWI, or wildfire concepts")
if q:
    req = Request(q, record=record or None, place=place or None, lat=lat, lon=lon, role=role)
    with st.spinner("Planning and gathering evidence..."):
        res = asyncio.run(orchestrator().ask(req))
    st.markdown(res.answer_markdown)
    for issue in res.issues:
        st.caption(f"⚠ {issue}")
    st.caption(f"intent {res.intent} · urgency {res.urgency} · {res.timings.get('total', 0):.1f}s · "
               f"{len(res.tool_calls)} tool calls · {sum((c['prompt_tokens'] or 0) + (c['completion_tokens'] or 0) for c in res.llm_calls)} tokens")

    pts = [h for e in res.evidence if e["kind"] == "live" and isinstance(e["data"], dict)
           for h in e["data"].get("hotspots", [])]
    if pts:
        df = pd.DataFrame([{"lat": h["latitude"], "lon": h["longitude"],
                            "color": "#d62728" if h["wildfire_evidence"] else "#7f7f7f"} for h in pts])
        if res.context.get("lat") is not None:
            df = pd.concat([df, pd.DataFrame([{"lat": res.context["lat"], "lon": res.context["lon"], "color": "#1f77b4"}])])
        st.map(df, latitude="lat", longitude="lon", color="color")
        st.caption("red: likely vegetation fire · grey: other thermal anomaly · blue: query location")

    with st.expander("Plan and steps"):
        for s in res.steps:
            st.write(f"**{s['id']} {s['agent']}** {'✅' if s['ok'] else '❌ ' + str(s['error'])}")
            for n in s["notes"]:
                st.caption(n)
    with st.expander("Evidence"):
        for e in res.evidence:
            st.markdown(f"**[{e['id']}]** {e['source']}" + (" · safety-critical" if e["safety_critical"] else ""))
            st.text(e["summary"])
            if e["query"]:
                st.code(e["query"], language="sparql")
            if isinstance(e["data"], list) and e["data"]:
                st.dataframe(pd.DataFrame(e["data"]))
