


from flask import Flask, render_template, session, request, redirect, url_for
import chess
import chess.svg
import os
import time
import helper

app = Flask(__name__)
app.config['SECRET_KEY'] = 'daf18fc78a01e813'
#IMAGE_DIR= 'C:/Users/mehta/Downloads/ArchiveFINAL'
IMAGE_DIR = 'C:/Users/mehta/Downloads/chessmoves'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_move(board, src_dest):
    try:
        move_obj = chess.Move.from_uci(src_dest[0] + src_dest[1])
        if move_obj in board.legal_moves:
            board.push(move_obj)
            return True
    except ValueError:
        pass

    try:
        move_obj_reversed = chess.Move.from_uci(src_dest[1] + src_dest[0])
        if move_obj_reversed in board.legal_moves:
            board.push(move_obj_reversed)
            return True
    except ValueError:
        pass

    return False

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'board' not in session or 'initialize' in request.form:
        session['board'] = chess.Board().fen()
        session['index'] = 0
        session['rotate'] = request.form.get('rotate') == 'on'
        files = sorted([f for f in os.listdir(IMAGE_DIR) if allowed_file(f)])
        session['files'] = files if files else []

    board = chess.Board(session['board'])
    files = session.get('files', [])
    index = session.get('index', 0)
    rotate_value = session.get('rotate', False)

    if request.method == 'POST':
        if 'move' in request.form and len(files) > index + 1:
            start_time = time.time()
            image_path_1 = os.path.join(IMAGE_DIR, files[index])
            image_path_2 = os.path.join(IMAGE_DIR, files[index + 1])
            move = helper.nextMove(image_path_1, image_path_2, rotate_value)
            if process_move(board, move):
                session['board'] = board.fen()
                session['index'] = (index + 1) % len(files)
                elapsed_time = time.time() - start_time
                session['elapsed_time'] = float('%.2f' % elapsed_time)
            else:
                print("Invalid move attempted")

    return render_template('index.html', board_svg=chess.svg.board(board), fen=board.fen(), 
                           elapsed_time=session.get('elapsed_time'), rotate=session.get('rotate', False))

@app.route('/reset-stock', methods=['POST'])
def reset_stock():
    session['board'] = chess.Board().fen()
    session['index'] = 0
    session.pop('elapsed_time', None)
    session['rotate'] = False  # Reset rotation to default when board is reset
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)


