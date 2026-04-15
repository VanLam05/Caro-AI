import pygame 
import sys
import os
import random

from .board import Board
from models.agentRandom import AgentRandom
from .buttons import Button


# ====================== CONSTANTS ======================
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (77, 199, 61)
RED = (199, 36, 55)
BLUE = (68, 132, 222)
GRAY = (100, 100, 100)

# Game states - simplified to just SETUP and PLAYING
SETUP = 0
PLAYING = 1
GAME_OVER = 2

# Game values
PLAYER_HUMAN = 1
PLAYER_AI = -1

# Board config
ROWNUM = 15
COLNUM = 15
winning_condition = 5
FPS = 120


# ====================== SETUP PYGAME ======================
pygame.init()
Window_size = [1280, 720]

# Calculate cell size
my_len_min = min(900/COLNUM, 720/ROWNUM)
MARGIN = my_len_min / 15
my_len_min = min((900 - MARGIN)/COLNUM, (720 - MARGIN) / ROWNUM)
my_len_min = my_len_min - MARGIN
WIDTH = my_len_min
HEIGHT = my_len_min

Screen = pygame.display.set_mode(Window_size)
clock = pygame.time.Clock()

# ====================== LOAD ASSETS ======================
path = os.path.join(os.getcwd(), 'asset')

# Piece images
x_img = pygame.transform.smoothscale(
    pygame.image.load(path + "/X_caro.png").convert_alpha(), 
    (my_len_min, my_len_min)
)
o_img = pygame.transform.smoothscale(
    pygame.image.load(path + "/O_caro.png").convert_alpha(), 
    (my_len_min, my_len_min)
)

# Button images
exit_img = pygame.transform.smoothscale(
    pygame.image.load(path + '/exit_btn.png').convert_alpha(), 
    (240, 105)
)
replay_img = pygame.transform.smoothscale(
    pygame.image.load(path + '/replay_btn.png').convert_alpha(), 
    (240, 105)
)
undo_img = pygame.transform.smoothscale(
    pygame.image.load(path + '/undo_btn.png').convert_alpha(), 
    (240, 105)
)
ai_img = pygame.transform.smoothscale(
    pygame.image.load(path + '/ai_btn.png').convert_alpha(), 
    (105, 105)
)
person_img = pygame.transform.smoothscale(
    pygame.image.load(path + '/person_btn.png').convert_alpha(), 
    (105, 105)
)
ai_img_gray = pygame.transform.smoothscale(
    pygame.image.load(path + '/ai_btn_gray.jpg').convert_alpha(), 
    (105, 105)
)
person_img_gray = pygame.transform.smoothscale(
    pygame.image.load(path + '/person_btn_gray.jpg').convert_alpha(), 
    (105, 105)
)
h_img = pygame.transform.smoothscale(
    pygame.image.load(path + '/h_btn.png').convert_alpha(), 
    (80, 80)
)
h_img_gray = pygame.transform.smoothscale(
    pygame.image.load(path + '/h_btn_gray.png').convert_alpha(), 
    (80, 80)
)
m_img = pygame.transform.smoothscale(
    pygame.image.load(path + '/m_btn.png').convert_alpha(), 
    (80, 80)
)
m_img_gray = pygame.transform.smoothscale(
    pygame.image.load(path + '/m_btn_gray.png').convert_alpha(), 
    (80, 80)
)
e_img = pygame.transform.smoothscale(
    pygame.image.load(path + '/e_btn.png').convert_alpha(), 
    (80, 80)
)
e_img_gray = pygame.transform.smoothscale(
    pygame.image.load(path + '/e_btn_gray.png').convert_alpha(), 
    (80, 80)
)
pvp_img = pygame.transform.smoothscale(
    pygame.image.load(path + '/player_vs_player.jpg').convert_alpha(), 
    (105, 105)
)
pvp_img_gray = pygame.transform.smoothscale(
    pygame.image.load(path + '/player_vs_player_gray.jpg').convert_alpha(), 
    (105, 105)
)
aivp_img = pygame.transform.smoothscale(
    pygame.image.load(path + '/ai_vs_player.jpg').convert_alpha(), 
    (105, 105)
)
aivp_img_gray = pygame.transform.smoothscale(
    pygame.image.load(path + '/ai_vs_player_gray.jpg').convert_alpha(), 
    (105, 105)
)

ai_thinking_img = pygame.transform.smoothscale(
    pygame.image.load(path + '/ai_thinking.png').convert_alpha(), 
    (105, 105)
)
ai_thinking_img_gray = pygame.transform.smoothscale(
    pygame.image.load(path + '/ai_thinking_gray.png').convert_alpha(), 
    (105, 105)
)

icon_img = pygame.transform.smoothscale(
    pygame.image.load(path + '/old/icon.jpg').convert_alpha(), 
    (20, 20)
)

# ====================== CREATE BUTTONS ======================
pvp_btn = Button(1075, 145, pvp_img, pvp_img_gray, 0.8)
aivp_btn = Button(970, 145, aivp_img, aivp_img_gray, 0.8)
person_btn = Button(1075, 305, person_img, person_img_gray, 0.8)
ai_btn = Button(970, 305, ai_img, ai_img_gray, 0.8)
h_btn = Button(1100, 235, h_img, h_img_gray, 0.8)
m_btn = Button(1035, 235, m_img, m_img_gray, 0.8)
e_btn = Button(970, 235, e_img, e_img_gray, 0.8)
undo_button = Button(970, 395, undo_img, undo_img, 0.8)
replay_button = Button(970, 485, replay_img, replay_img, 0.8)
exit_button = Button(970, 575, exit_img, exit_img, 0.8)
ai_thinking_btn = Button(1020, 30, ai_thinking_img, ai_thinking_img_gray, 0.8)

pygame.display.set_caption('Caro AI')
pygame.display.set_icon(icon_img)


# ====================== GLOBAL STATE ======================
class GameState:
    def __init__(self):
        self.state = SETUP  # SETUP or PLAYING or GAME_OVER
        self.board = Board(rows=ROWNUM, cols=COLNUM, winning_condition=winning_condition)
        self.is_pvp = True  # True for PvP, False for PvAI
        self.difficulty = 0  # 0=easy, 1=medium, 2=hard
        self.human_first = True  # True if human goes first
        self.agent = None
        self.winner = 0  # 0=none, 1=human/X, -1=AI/O, 2=draw
        self.ai_thinking = False
        self.game_over = False
        self.game_started = False  # True if game has started
        self.ai_think_start_time = None  # Track when AI started thinking
        self.ai_thinking_duration = 500  # milliseconds to wait while thinking
    
    def can_setup(self):
        """Check if we can still change setup (no moves made yet)"""
        return len(self.board.move_history) == 0
    
    def reset_for_new_game(self):
        """Reset board but keep settings"""
        self.board.reset()
        self.game_over = False
        self.winner = 0
        self.ai_thinking = False
        if self.is_pvp:
            self.board.turn = PLAYER_HUMAN
        elif self.human_first:
            self.board.turn = PLAYER_HUMAN
        else:
            self.board.turn = PLAYER_AI


game = GameState()


# ====================== UTILITY FUNCTIONS ======================
# def draw_logo():
#     """Draw logo text"""
#     font = pygame.font.Font('freesansbold.ttf', 36)
#     text = font.render('By nhóm 2', True, WHITE, BLACK)
#     textRect = text.get_rect()
#     textRect.center = (1100, 700)
#     Screen.blit(text, textRect)


def draw_board():
    """Draw game board with pieces"""
    # draw_logo()
    for row in range(ROWNUM):
        for col in range(COLNUM):
            color = WHITE
            # Highlight last move
            if len(game.board.move_history) > 0:
                last_move_row, last_move_col = game.board.move_history[-1][0], game.board.move_history[-1][1]
                if row == last_move_row and col == last_move_col:
                    color = GREEN
            
            pygame.draw.rect(
                Screen,
                color,
                [(MARGIN + WIDTH) * col + MARGIN,
                 (MARGIN + HEIGHT) * row + MARGIN,
                 WIDTH,
                 HEIGHT]
            )
            
            # Draw pieces
            if game.board.grid[row][col] == 1:
                Screen.blit(x_img, ((WIDTH + MARGIN) * col + MARGIN, (HEIGHT + MARGIN) * row + MARGIN))
            elif game.board.grid[row][col] == -1:
                Screen.blit(o_img, ((WIDTH + MARGIN) * col + MARGIN, (HEIGHT + MARGIN) * row + MARGIN))


def draw_text_centered(text, font_size, color, bg_color=None):
    """Draw centered text on screen"""
    font = pygame.font.Font('freesansbold.ttf', font_size)
    text_surf = font.render(text, True, color, bg_color)
    text_rect = text_surf.get_rect()
    text_rect.center = (Window_size[0] / 2, Window_size[1] / 2)
    Screen.blit(text_surf, text_rect)


def draw_game_over_screen():
    """Draw game over message overlay"""
    if game.is_pvp:
        # PvP mode - show player numbers
        if game.winner == 1:
            draw_text_centered('Player 1 (X) Wins!', 100, RED, BLUE)
        elif game.winner == -1:
            draw_text_centered('Player 2 (O) Wins!', 100, BLUE, RED)
        elif game.winner == 2:
            draw_text_centered('Draw!', 100, GREEN, BLUE)
    else:
        # PvAI mode - show Human vs AI
        if game.winner == 1:
            draw_text_centered('You (X) Win!', 100, RED, BLUE)
        elif game.winner == -1:
            draw_text_centered('AI (O) Wins!', 100, BLUE, RED)
        elif game.winner == 2:
            draw_text_centered('Draw!', 100, GREEN, BLUE)


def make_ai_move():
    """Make AI move with non-blocking timer"""
    if not game.agent or game.board.turn != PLAYER_AI or game.game_over:
        return
    
    # Start thinking timer if not already started
    if not game.ai_thinking:
        game.ai_thinking = True
        game.ai_think_start_time = pygame.time.get_ticks()
        return
    
    # Check if enough time has passed
    current_time = pygame.time.get_ticks()
    elapsed = current_time - game.ai_think_start_time
    
    if elapsed < game.ai_thinking_duration:
        # Still thinking, don't make move yet
        return
    
    # Time's up, make the move
    move = game.agent.get_move()
    if move:
        game.board.make_move(move[0], move[1])
        game.winner = game.board.get_winner()
        if game.winner != 0:
            game.game_over = True
            game.state = GAME_OVER
    
    game.ai_thinking = False
    game.ai_think_start_time = None


def undo_move():
    """Undo last move"""
    if len(game.board.move_history) == 0:
        return
    
    if game.is_pvp:
        # PvP: undo 1 move
        game.board.undo_move()
    else:
        # PvAI: undo 2 moves (human move + AI move)
        if len(game.board.move_history) >= 2:
            game.board.undo_move()
            game.board.undo_move()
        elif len(game.board.move_history) == 1:
            game.board.undo_move()
    
    game.winner = 0
    game.game_over = False
    game.ai_thinking = False
    game.ai_think_start_time = None



# ====================== UTILITY FUNCTIONS ======================
def reset_buttons():
    """Reset all buttons to default state"""
    pvp_btn.disable_button()
    aivp_btn.disable_button()
    person_btn.disable_button()
    ai_btn.disable_button()
    h_btn.disable_button()
    m_btn.disable_button()
    e_btn.disable_button()


def update_button_states():
    """Update button enable/disable states based on game state"""
    if not game.can_setup():
        # Game has started - disable setup buttons
        pvp_btn.disable_button()
        aivp_btn.disable_button()
        person_btn.disable_button()
        ai_btn.disable_button()
        h_btn.disable_button()
        m_btn.disable_button()
        e_btn.disable_button()
    else:
        # Game hasn't started - enable mode selection
        pvp_btn.enable_button()
        aivp_btn.enable_button()
        
        if game.is_pvp:
            # PvP mode selected
            pvp_btn.disable_button()
            aivp_btn.enable_button()
            person_btn.disable_button()
            ai_btn.disable_button()
            h_btn.disable_button()
            m_btn.disable_button()
            e_btn.disable_button()
        else:
            # PvAI mode selected
            aivp_btn.disable_button()
            pvp_btn.enable_button()
            
            # Difficulty buttons
            if game.difficulty == 0:
                e_btn.disable_button()
                m_btn.enable_button()
                h_btn.enable_button()
            elif game.difficulty == 1:
                e_btn.enable_button()
                m_btn.disable_button()
                h_btn.enable_button()
            elif game.difficulty == 2:
                e_btn.enable_button()
                m_btn.enable_button()
                h_btn.disable_button()
            
            # First player buttons
            if game.human_first:
                person_btn.disable_button()
                ai_btn.enable_button()
            else:
                person_btn.enable_button()
                ai_btn.disable_button()


def handle_setup_state(event):
    """Handle setup state - mode selection"""
    # Mode buttons
    if pvp_btn.draw(Screen):
        if game.can_setup():
            game.is_pvp = True
            update_button_states()
    
    if aivp_btn.draw(Screen):
        if game.can_setup():
            game.is_pvp = False
            update_button_states()
    
    # Difficulty buttons (only for PvAI)
    if not game.is_pvp and game.can_setup():
        if e_btn.draw(Screen):
            game.difficulty = 0
            update_button_states()
        
        if m_btn.draw(Screen):
            game.difficulty = 1
            update_button_states()
        
        if h_btn.draw(Screen):
            game.difficulty = 2
            update_button_states()
        
        # First player buttons
        if person_btn.draw(Screen):
            game.human_first = True
            update_button_states()
        
        if ai_btn.draw(Screen):
            game.human_first = False
            update_button_states()


def handle_playing_state(event):
    """Handle playing state - game interaction"""
    # Undo button
    if undo_button.draw(Screen):
        undo_move()
    
    # Replay button - reset board but allow mode re-selection
    if replay_button.draw(Screen):
        game.board.reset()
        game.game_started = False
        game.ai_thinking = False
        game.ai_think_start_time = None
        game.state = SETUP
        update_button_states()
        return None
    
    # Mouse click for placing piece
    if event.type == pygame.MOUSEBUTTONDOWN and not game.game_over and not game.ai_thinking:
        pos = pygame.mouse.get_pos()
        col = int(pos[0] // (WIDTH + MARGIN))
        row = int(pos[1] // (HEIGHT + MARGIN))
        
        if col < COLNUM and row < ROWNUM:
            # Check if it's playable turn
            if (game.is_pvp and True) or (not game.is_pvp and game.board.turn == PLAYER_HUMAN):
                if game.board.make_move(row, col):
                    game.winner = game.board.get_winner()
                    if game.winner != 0:
                        game.game_over = True
                        game.state = GAME_OVER
    
    return None


def draw_current_state():
    """Draw UI - always show board with buttons"""
    Screen.fill(BLACK)
    
    # Always draw board and buttons
    draw_board()
    
    # Draw control buttons
    undo_button.draw(Screen)
    exit_button.draw(Screen)
    replay_button.draw(Screen)
    
    # Draw AI thinking button in PvAI mode
    if not game.is_pvp:
        ai_thinking_btn.draw(Screen)
    
    # Draw mode and difficulty buttons
    pvp_btn.draw(Screen)
    aivp_btn.draw(Screen)
    
    # Draw AI specific buttons
    if not game.is_pvp:
        person_btn.draw(Screen)
        ai_btn.draw(Screen)
        e_btn.draw(Screen)
        m_btn.draw(Screen)
        h_btn.draw(Screen)
    
    # Draw game over message if game ended
    if game.game_over:
        draw_game_over_screen()


# ====================== MAIN LOOP ======================
def main():
    """Main game loop"""
    global game
    
    reset_buttons()
    update_button_states()
    done = False
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            # Check exit button globally (works from any state)
            if exit_button.draw(Screen):
                done = True
            
            # Handle events based on state
            if game.state == SETUP:
                handle_setup_state(event)
                
                # Check for board click to start game from SETUP
                if event.type == pygame.MOUSEBUTTONDOWN and game.can_setup():
                    pos = pygame.mouse.get_pos()
                    col = int(pos[0] // (WIDTH + MARGIN))
                    row = int(pos[1] // (HEIGHT + MARGIN))
                    
                    if col < COLNUM and row < ROWNUM:
                        # Click on board starts the game
                        game.state = PLAYING
                        game.game_started = True
                        if not game.is_pvp:
                            game.agent = AgentRandom(game.board)
                        update_button_states()
                        if game.board.make_move(row, col):
                            game.winner = game.board.get_winner()
                            if game.winner != 0:
                                game.game_over = True
                                game.state = GAME_OVER
            
            elif game.state == PLAYING:
                result = handle_playing_state(event)
                if result == "EXIT":
                    done = True
            
            elif game.state == GAME_OVER:
                # Undo button - go back to previous move
                if undo_button.draw(Screen):
                    undo_move()
                    game.state = PLAYING
                
                # Replay button - reset and stay in SETUP to allow mode re-selection
                if replay_button.draw(Screen):
                    game.board.reset()
                    game.game_started = False
                    game.game_over = False
                    game.winner = 0
                    game.ai_thinking = False
                    game.ai_think_start_time = None
                    game.state = SETUP
                    update_button_states()
        
        # Update AI thinking button state before drawing
        if game.state == PLAYING and not game.is_pvp:
            if game.ai_thinking:
                ai_thinking_btn.disable_button()  # Bright
            else:
                ai_thinking_btn.enable_button()  # Gray
        
        # AI turn in playing state
        if game.state == PLAYING and not game.is_pvp and not game.game_over and game.board.turn == PLAYER_AI:
            make_ai_move()
        
        # Draw current state
        draw_current_state()
        
        # Update display
        pygame.display.update()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()