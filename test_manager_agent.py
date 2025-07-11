#!/usr/bin/env python3
"""
Test script for querying the manager agent via the MCP server.
"""
import requests
import sys
from typing import Iterator


def query_manager_agent(
    query: str,
    base_url: str = "http://localhost:5055",
    session_id: str = None
) -> Iterator[str]:
    """
    Query the manager agent with a given query.
    
    Args:
        query: The query to send to the manager agent
        base_url: Base URL of the MCP server
        session_id: Optional session ID for the request
    
    Yields:
        Response chunks from the streaming response
    """
    url = f"{base_url}/agents/manager"
    
    payload = {
        "query": query,
        "stream": True
    }
    
    if session_id:
        payload["session_id"] = session_id
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=30
        )
        response.raise_for_status()
        
        print(f"✓ Successfully connected to {url}")
        print(f"✓ Query: {query}")
        print("✓ Response:")
        print("-" * 50)
        
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                print(chunk, end="", flush=True)
                yield chunk
                
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to {url}")
        print("  Make sure the MCP server is running on the specified host and port.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("✗ Error: Request timed out after 30 seconds")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP Error: {e}")
        print(f"  Response: {response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)


def main():
    """Main function to run the test."""
    print("CodeArkt Manager Agent Test")
    print("=" * 40)
    
    # The specific query from the user
    query = "Call librarian agent to find an abstract of the PingPong paper by Ilya Gusev"
    
    # Query the manager agent
    responses = list(query_manager_agent(query))
    
    print("\n" + "-" * 50)
    print("✓ Test completed successfully!")
    print(f"✓ Received {len(responses)} response chunks")


if __name__ == "__main__":
    main() 