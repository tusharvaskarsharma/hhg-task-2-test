from backend.pipeline.tokenizer import preprocess_query

def test_preprocess_query():
    # leading/trailing spaces
    assert preprocess_query("  hello world  ") == "hello world"
    
    # repeated spaces
    assert preprocess_query("hello    world") == "hello world"
    
    # unicode preservation
    assert preprocess_query("भारत की राजधानी") == "भारत की राजधानी"
    
    # english
    assert preprocess_query("what is the capital of india") == "what is the capital of india"
