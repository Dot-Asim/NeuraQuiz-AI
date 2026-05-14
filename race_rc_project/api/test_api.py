import time
import requests
import subprocess
import os


def run_tests():
    # Start the server
    print("Starting API server with dotasim environment...")
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'src'))
    server = subprocess.Popen(
        ['python', '-m', 'uvicorn', 'api.main:app', '--port', '8124'],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        env=env
    )
    time.sleep(15)  # Wait for models to load (they take some time)

    url = "http://localhost:8124/api"
    results = {}

    try:
        # 1. Random article
        print("Testing /api/random_article...")
        r = requests.get(f"{url}/random_article")
        results["random_article_status"] = r.status_code
        if r.status_code == 200:
            data = r.json()
            results["random_article_success"] = True

            # 2. Quiz Gen
            print("Testing /api/quiz...")
            quiz_payload = {
                "article": data["article"],
                "question": data["question"],
                "correct_answer": data["correct_answer"],
            }
            r2 = requests.post(f"{url}/quiz", json=quiz_payload)
            results["quiz_gen_status"] = r2.status_code
            if r2.status_code == 200:
                results["quiz_gen_success"] = True
                r2.json()
            else:
                results["quiz_gen_success"] = False
                print(r2.text)

            # 3. Verify Ans
            print("Testing /api/verify...")
            verify_payload = {
                "article": data["article"],
                "question": data["question"],
                "selected_option": data["correct_answer"],
            }
            r3 = requests.post(f"{url}/verify", json=verify_payload)
            results["verify_status"] = r3.status_code
            if r3.status_code == 200:
                results["verify_success"] = True
                r3.json()
            else:
                results["verify_success"] = False
                print(r3.text)
        else:
            results["random_article_success"] = False
    except Exception as e:
        print("Error during tests:", str(e))
    finally:
        print("Terminating server...")
        server.terminate()
        server.wait()

    print("\n--- TEST SUMMARY ---")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    run_tests()
