import pygame


class GamePadAgent:

    def action(self):
        input_val = self.get()
        x_input = input_val[0]
        y_input = input_val[1]
        z_input = input_val[2]
        r_x_input = input_val[3]
        r_y_input = input_val[4]
        r_z_input = input_val[5]
        reset = input_val[6]
        pause = input_val[7]
        act = [
            y_input,
            x_input,
            r_x_input,
            r_z_input]
        return act

    def get(self):
        pygame.init()
        j = pygame.joystick.Joystick(0)
        j.init()
        out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        it = 0  # iterator
        pygame.event.pump()

        # Read input from the two joysticks
        for i in range(0, j.get_numaxes()):
            out[it] = j.get_axis(i)
            it += 1
        # Read input from buttons
        for i in range(0, j.get_numbuttons()):
            out[it] = j.get_button(i)
            it += 1
        return out
