```markdown
# Tic-Tac-Toe Frame Component

This document provides a technical overview of the `Frame.java` file, which is responsible for creating and managing the main window of the Tic-Tac-Toe game application.

## Overview

The `Frame` class extends `JPanel` and utilizes `JFrame` to create the main window for the Tic-Tac-Toe game. It sets up the window's dimensions, title, close operation, and adds the game panel to it.

## File: `Frame.java`

### Imports

*   `java.awt.Dimension`: Used to define the dimensions (width and height) of the window.
*   `javax.swing.JPanel`:  The base class for the `Frame` class, allowing it to be added to the `JFrame`.
*   `javax.swing.JFrame`: Provides the main window for the application.

### Class Declaration

```java
public class Frame extends JPanel {
    // ...
}
```

*   Declares the `Frame` class, which inherits from `JPanel`. This makes the `Frame` itself a panel that can be added to the `JFrame`.

### Class Members (Constants)

*   `public static final int HEIGHT = 800;`: Defines the height of the game window as 800 pixels.  `static` and `final` means this value is constant and shared across all instances of the `Frame` class.
*   `public static final int WIDTH = 800;`: Defines the width of the game window as 800 pixels. `static` and `final` means this value is constant and shared across all instances of the `Frame` class.
*   `public static final String TITLE = "Tic-Tac-Toe";`: Defines the title of the game window as "Tic-Tac-Toe".  `static` and `final` means this value is constant and shared across all instances of the `Frame` class.

### Class Members (Instance Variables)

*   `private JFrame frame;`:  A `JFrame` object that represents the main window.  It's declared as `private` to encapsulate the `JFrame` object within the `Frame` class.

### Constructor

```java
public Frame(Game game) {
    frame = new JFrame(TITLE);
    frame.setPreferredSize(new Dimension(WIDTH, HEIGHT));
    frame.setMaximumSize(new Dimension(WIDTH, HEIGHT));
    frame.setMinimumSize(new Dimension(WIDTH, HEIGHT));
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.setLocationRelativeTo(null);
    frame.pack();
    frame.add(game);
    frame.setVisible(true);
    game.start();
}
```

*   **`public Frame(Game game)`**:  The constructor for the `Frame` class. It accepts a `Game` object as a parameter, which represents the game logic and UI.

    *   **`frame = new JFrame(TITLE);`**: Creates a new `JFrame` object with the title "Tic-Tac-Toe".
    *   **`frame.setPreferredSize(new Dimension(WIDTH, HEIGHT));`**: Sets the preferred size of the frame to 800x800 pixels.
    *   **`frame.setMaximumSize(new Dimension(WIDTH, HEIGHT));`**: Sets the maximum size of the frame to 800x800 pixels.  This and the previous line can prevent the user from resizing the window.
    *   **`frame.setMinimumSize(new Dimension(WIDTH, HEIGHT));`**: Sets the minimum size of the frame to 800x800 pixels. This and the previous two lines effectively make the window unresizable.
    *   **`frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);`**: Sets the default close operation to exit the application when the window is closed.
    *   **`frame.setLocationRelativeTo(null);`**: Centers the window on the screen.
    *   **`frame.pack();`**: Sizes the frame so that all its contents are at or above their preferred sizes.
    *   **`frame.add(game);`**: Adds the `Game` panel to the frame, making it visible within the window.
    *   **`frame.setVisible(true);`**: Makes the frame visible to the user.
    *   **`game.start();`**: Calls the `start()` method of the `Game` object, presumably to initialize and start the game logic.

## Functionality

The `Frame` class is responsible for:

*   Creating the main application window.
*   Setting the window's size and title.
*   Centering the window on the screen.
*   Adding the `Game` panel to the window, displaying the Tic-Tac-Toe board.
*   Starting the game logic.
*   Handling the window close operation (exiting the application).

## Dependencies

*   `Game.java` (The game logic and UI component)

## Usage

To use the `Frame` class, you need to:

1.  Create an instance of the `Game` class.
2.  Create an instance of the `Frame` class, passing the `Game` object to the constructor.  This will create and display the game window.

```java
Game game = new Game();
Frame frame = new Frame(game);
```

## Notes

*   The `Frame` class sets both the maximum and minimum sizes of the window, effectively preventing the user from resizing it.  If resizing is desired, these lines should be removed or modified.
*   The `game.start()` method call suggests that the `Game` class has a method to initialize and begin the game.
