import contextlib, inspect, io, json, logging, os, sys
sys.path.insert(0, os.getcwd())
from pathlib import Path
from unittest.mock import patch
os.environ['ACADEMICFORGE_FRONTEND_MODE']='cloud_demo'
os.environ['HF_HUB_OFFLINE']='1'
from frontend import cloud_demo_runtime as demo
from backend.sample_data import get_sample_papers
from frontend.api_client import APIClient
from streamlit.testing.v1 import AppTest
result={}
from importlib.metadata import version
result['python']=sys.version.split()[0]
result['packages']={name:version(name) for name in ['streamlit','arxiv','openai','requests','numpy','pytest']}
result['method']='AppTest with controlled source fixtures; no live academic or paid generation calls'
result['optional_dependencies']={}
for module in ['sentence_transformers','torch','fastapi','dotenv']:
    import importlib.util
    result['optional_dependencies'][module]=importlib.util.find_spec(module) is not None
logs=io.StringIO()
h=logging.StreamHandler(logs)
logging.getLogger().addHandler(h)
app=AppTest.from_file('frontend/streamlit_cloud_app.py').run(timeout=20)
result['startup_exceptions']=[str(e.value) for e in app.exception]
result['inputs']=[x.label for x in app.text_input]
result['selectboxes']=[x.label for x in app.selectbox]
result['runtime_retrieve_signature']=str(inspect.signature(demo.retrieve_papers))
query='retrieval augmented generation hallucinations'
with patch.object(demo,'search_live_candidates',return_value=([],False)):
    response=demo.retrieve_papers(query)
    result['source_outage']={'paper_count':len(response['papers']), 'sources':sorted({p['source'] for p in response['papers']}),'message':response['message']}
    app.text_input[0].set_value(query)
    next(b for b in app.button if b.label=='Search').click().run(timeout=20)
    result['outage_search_ui_exceptions']=[str(e.value) for e in app.exception]
    result['outage_search_ui_errors']=[x.value for x in app.error]
    result['buttons']=[x.label for x in app.button]
next(b for b in app.button if b.label=='Generate Research Plan').click().run(timeout=30)
result['plan_ui_exceptions']=[str(e.value) for e in app.exception]
result['plan_ui_errors']=[x.value for x in app.error]
result['plan_created']=bool(app.session_state['research_plan'])
next(b for b in app.button if b.label=='Generate Research Plan').click().run(timeout=30)
result['second_plan_ui_errors']=[x.value for x in app.error]
paper=get_sample_papers()[0]
paper['source']='arxiv'
paper['paper_id']='real-paper-fixture'
with patch.object(demo,'search_live_candidates',return_value=([paper],False)):
    response=demo.retrieve_papers(query)
    result['one_live_match']={'sources':sorted({p['source'] for p in response['papers']}),'retains_live_paper':any(p['paper_id']=='real-paper-fixture' for p in response['papers'])}
with patch.object(demo,'search_live_candidates',return_value=([],True)):
    result['missing_direct_id']=demo.retrieve_papers('9999.99999')
client=APIClient()
with patch.dict(os.environ,{'ACADEMICFORGE_CLOUD_ENABLE_LLM':'true'}):
    result['enable_llm_provider']=client.get_config()['llm_provider']
    result['enable_llm_summary']=client.post_json('/summarize',paper)['summary'][:100]
result['shared_identifier_without_headers']=client._client_identifier()
result['logs']=logs.getvalue()
print(json.dumps(result,indent=2))
Path('/tmp/academicforge-demo-audit-results.json').write_text(json.dumps(result,indent=2))
