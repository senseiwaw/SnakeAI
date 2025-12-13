import arcade



WINDOW_WIDTH  = 900	
WINDOW_HEIGHT = 600
WINDOW_TITLE  = "Snake"

class GameView(arcade.Window):
    """
    Main application class.
    """

    def __init__(self):

        # Call the parent class to set up the window
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE,resizable =True)
        self.background_color = arcade.color.CHARTREUSE
        self.player_texture = arcade.load_texture(":resources:images/animated_characters/female_adventurer/femaleAdventurer_idle.png")
        self.player_sprite = arcade.Sprite(self.player_texture)
        self.player_sprite.center_x = 300
        self.player_sprite.center_y = 200
        self.player_list = arcade.SpriteList()
        self.player_list.append(self.player_sprite)

    def setup(self):
        """Set up the game here. Call this function to restart the game."""
        pass

    def on_draw(self):
        """Render the screen."""

        # The clear method should always be called at the start of on_draw.
        # It clears the whole screen to whatever the background color is
        # set to. This ensures that you have a clean slate for drawing each
        # frame of the game.
        self.clear()

        # Code to draw other things will go here
        self.player_list.draw()

def main():
    """Main function"""
    window = GameView()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
