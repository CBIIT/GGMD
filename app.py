from flask import Flask, request, Response, session, redirect, url_for
import subprocess
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(24)

def send_email(recipient, subject, body):
    """Send email using sendmail."""
    # Construct the email message
    message = f"Subject: {subject}\nTo: {recipient}\n{body}"

    try:
        # Send the message using sendmail
        p = subprocess.Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=subprocess.PIPE)
        p.communicate(message.encode('utf8'))
        print("Email sent successfully!")
    except Exception as e:
        # Print any error messages to stdout
        print(f"Failed to send email: {e}")

def run_script_and_stream_output(initial_pop_size, max_population):
    cmd = ['bash', '/submit.sh', initial_pop_size, max_population]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    with open('/mnt/results.log', 'a') as file:
        for line in iter(process.stdout.readline, ''):
            file.write(line)
            yield line
    #send_email("jingczhang@microsoft.com", "AML Job Complete", "Your Azure machine learning GMD job is completed")

@app.route('/', methods=['GET'])
def start():
    return redirect(url_for('get_initial_pop_size'))

@app.route('/initial_pop_size', methods=['GET', 'POST'])
def get_initial_pop_size():
    if request.method == 'POST':
        session['initial_pop_size'] = request.form['initial_pop_size']
        return redirect(url_for('get_max_population'))
    else:
        return '''
                <form method="post">
                    Initial Population Size: <input type="text" name="initial_pop_size"><br>
                    <input type="submit" value="Next">
                </form>
            '''

@app.route('/max_population', methods=['GET', 'POST'])
def get_max_population():
    if request.method == 'POST':
        session['max_population'] = request.form['max_population']
        return redirect(url_for('run_simulation'))
    else:
        return '''
                <form method="post">
                    Max Population: <input type="text" name="max_population"><br>
                    <input type="submit" value="Submit">
                </form>
            '''

@app.route('/run', methods=['GET'])
def run_simulation():
    initial_pop_size = session.get('initial_pop_size', '')
    max_population = session.get('max_population', '')

    # Stream output of the script to the response
    return Response(run_script_and_stream_output(initial_pop_size, max_population), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8787)

