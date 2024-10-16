from manim import *

class NeuralNetworkAnimation(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        Text.set_default(font="Roboto", color=BLACK)
        
        # Show the Dwengo logo in the center of the screen
        self.show_dwengo_logo()
        
        
        # Title of the animation
        title = Text("Wat is 'transfer learning'?", font="Roboto").scale(0.8)
        # move the title to the top of the screen with a small margin
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))
        
        
        # Create layers
        input_layer = self.create_layer(3, LEFT*4, add_ellipsis=True)
        hidden_layer = self.create_layer(4, LEFT*2)
        hidden_layer2 = self.create_layer(4, ORIGIN)
        hidden_layer3 = self.create_layer(7, RIGHT*2)
        output_layer = self.create_layer(2, RIGHT*4, color=ManimColor("#fab63c"))
        
        layers = [input_layer, hidden_layer, hidden_layer2, hidden_layer3]

        # Animate network layers
        self.animate_layers(layers)
        
        # Add labels
        labels = self.add_labels(layers[-1], ["krant", "Blik", "Fles", "zak", "folie", "flyer", "doos"])
        
        # Get the layer values from the paper
        layer_values = self.get_layer_values_paper()

        # Animate the predictions
        self.animate_predictions(layers, layer_values, ".\\nn\\images\\krant2.png")
        
        # Get the layer values from folie
        layer_values = self.get_layer_values_zak()
        
        # Animate the predictions
        self.animate_predictions(layers, layer_values, ".\\nn\\images\\zak.png")
        
        # Wait for the animation to finish
        self.wait(2)
        
        # Remove labels
        self.play(FadeOut(*labels))
        
        # Add last layer
        self.play(Create(output_layer))

        # connect the layers with edges
        self.connect_layers(layers[-1], output_layer, has_first_layer=False, color=ManimColor("#fab63c"))
        
        # Add labels
        labels = self.add_labels(output_layer, ["Papier", "PMD"])
        
        self.wait(3)
        
        # Call the function to create and display the stack of abstract images
        self.show_input_image(".\\nn\\images\\image_stack.png")
        self.wait(2)
        
        # Animate the predictions
        training_results = [np.array([4, 4]), np.array([4, 3]), np.array([2, 5]), np.array([1, 6]), np.array([8, 1])]
        backprop_results= [np.array([1, 8, 9, 2, 1, 1, 1]), np.array([8, 8, 9, 8, 1, 1, 1]), np.array([7, 2, 2, 2, 1, 1, 10]), np.array([8, 12, 9, 2, 10, 11, 2]), np.array([6, 1, 4, 2, 1, 2, 10])]
        layers.append(output_layer)
        
        for train_result, backprop_result in zip(training_results, backprop_results):
            rand_layer_values = self.get_layer_values_ranom(fixed_output_layer=train_result)
            self.animate_predictions(layers, rand_layer_values)
            
            self.feed_images_with_proper_dots([output_layer, hidden_layer3], [rand_layer_values[-1], backprop_result])
            self.wait(2)
            
            
        #rand_layer_values = self.get_layer_values_ranom(fixed_output_layer=np.array([3, 3]))
        #self.animate_predictions(layers, rand_layer_values)
        
        #self.feed_images_with_proper_dots([output_layer, hidden_layer3], [rand_layer_values[-1], np.array([8, 2, 9, 2, 1, 1, 10])])
        
        self.wait(5)
        
               
    def show_dwengo_logo(self):
        # Show the Dwengo logo in the bottom right corner
        dwengo_logo = ImageMobject(".\\nn\\images\\dwengo-groen-zwart.png").scale(0.5)
        self.play(FadeIn(dwengo_logo))
        self.wait(3)
        self.play(FadeOut(dwengo_logo))
        
    def get_layer_values_paper(self):
        input_vector = np.array([2, 4, 8])
        w1 = np.array([[4, 1, 4, 1], [5, 3, 5, 1], [3, 2, 4, 1]])
        w2 = np.array([[5, 6, 1, 1], [6, 5, 2, 1], [5, 6, 1, 1], [7, 4, 3, 2]])
        w3 = np.array([[8, 2, 4, 3, 4, 1, 2], [9, 1, 4, 3, 1, 1, 1], [6, 3, 3, 2, 2, 2, 1], [5, 1, 2, 2, 1, 2, 1]])
        
        # normalize w1, w2, w3
        w1 = w1 / np.linalg.norm(w1)
        w2 = w2 / np.linalg.norm(w2)
        w3 = w3 / np.linalg.norm(w3)
        
        self.layer_weights = [w1, w2, w3]
        self.layer_biases = [np.random.rand(4), np.random.rand(4), np.random.rand(7)]
        # Create a ranomd input vector
        
        # propagate the vector through the network by multiplying with the weights and adding the biases
        
        hidden_layer = np.dot(input_vector, self.layer_weights[0]) + self.layer_biases[0]
        hidden_layer2 = np.dot(hidden_layer, self.layer_weights[1]) + self.layer_biases[1]
        output_layer = np.dot(hidden_layer2, self.layer_weights[2]) + self.layer_biases[2]
        
        layer_values = [input_vector, hidden_layer, hidden_layer2, output_layer]
        
        return layer_values
    
    def get_layer_values_ranom(self, fixed_output_layer=None, fixed_hidden_layer=None):
        # Get the layer values from the paper
        input_vector = np.array([2, 4, 8])
        w1 = np.random.rand(3, 4)
        w2 = np.random.rand(4, 4)
        w3 = np.random.rand(4, 7)
        w4 = np.random.rand(7, 2)
        
        # normalize w1, w2, w3
        w1 = w1 / np.linalg.norm(w1)
        w2 = w2 / np.linalg.norm(w2)
        w3 = w3 / np.linalg.norm(w3)
        w4 = w4 / np.linalg.norm(w4)
        
        self.layer_weights = [w1, w2, w3, w4]
        self.layer_biases = [np.random.rand(4), np.random.rand(4), np.random.rand(7), np.random.rand(2)]
        # Create a ranomd input vector
        
        # propagate the vector through the network by multiplying with the weights and adding the biases
        hidden_layer = np.dot(input_vector, self.layer_weights[0]) + self.layer_biases[0]
        hidden_layer2 = np.dot(hidden_layer, self.layer_weights[1]) + self.layer_biases[1]
        
        if fixed_hidden_layer is not None:
            hidden_layer3 = fixed_hidden_layer
        else:
            hidden_layer3 = np.dot(hidden_layer2, self.layer_weights[2]) + self.layer_biases[2]
        
        if fixed_output_layer is not None:
            output_layer = fixed_output_layer
        else:
            output_layer = np.dot(hidden_layer3, self.layer_weights[3]) + self.layer_biases[3]
        
        layer_values = [input_vector, hidden_layer, hidden_layer2, hidden_layer3, output_layer]
        
        return layer_values
    
    def get_layer_values_zak(self):
        # Get the layer values from the paper
        input_vector = np.array([8, 4, 8])
        w1 = np.array([[4, 1, 4, 1], [5, 3, 5, 1], [3, 2, 4, 1]])
        w2 = np.array([[5, 6, 1, 1], [6, 5, 2, 1], [5, 6, 1, 1], [7, 4, 3, 2]])
        w3 = np.array([[1, 2, 4, 8, 3, 1, 2], [1, 1, 4, 9, 3, 1, 1], [1, 3, 3, 6, 2, 2, 1], [2, 1, 2, 12, 1, 2, 1]])
        
        # normalize w1, w2, w3
        w1 = w1 / np.linalg.norm(w1)
        w2 = w2 / np.linalg.norm(w2)
        w3 = w3 / np.linalg.norm(w3)
        
        self.layer_weights = [w1, w2, w3]
        self.layer_biases = [np.random.rand(4), np.random.rand(4), np.random.rand(7)]
        # Create a ranomd input vector
        
        # propagate the vector through the network by multiplying with the weights and adding the biases
        hidden_layer = np.dot(input_vector, self.layer_weights[0]) + self.layer_biases[0]
        hidden_layer2 = np.dot(hidden_layer, self.layer_weights[1]) + self.layer_biases[1]
        output_layer = np.dot(hidden_layer2, self.layer_weights[2]) + self.layer_biases[2]
        
        layer_values = [input_vector, hidden_layer, hidden_layer2, output_layer]
        
        return layer_values
    
    def show_input_image(self, image_path):
        # Simulate feeding images
        input_image = ImageMobject(image_path).scale(0.5)
        input_image.to_edge(LEFT).shift(DOWN*0.5)
        self.play(FadeIn(input_image))
        self.wait(1)
        return input_image
        
    def animate_predictions(self, layers, layer_values, image_path=None):
        if image_path:
            # Simulate feeding images
            input_image = self.show_input_image(image_path)
        
        # show progression through the network
        self.feed_images_with_proper_dots(layers, layer_values)
        
        #show_dots_at_layer
        dots = self.show_dots_at_layer(layers[-1], layer_values[-1])

        # Show classification result
        self.show_output_vector(layers[-1], layer_values[-1])
        
        # Remove the dots and input image
        if image_path:
            self.play(FadeOut(input_image))
        for dot in dots:
            self.remove(dot)
                

    def create_layer(self, num_nodes, position, add_ellipsis=False, color=ManimColor("#0e6942")):
        # Create a layer of circles (neurons)
        layer = VGroup()
        for i in range(num_nodes):
            neuron = Circle(radius=0.3, color=color)
            neuron.shift(UP*(i - num_nodes/2) + position)
            layer.add(neuron)
            
        # Add ellipsis dots if it's the input layer to imply more nodes
        if add_ellipsis:
            ellipsis_above = VGroup(*[Dot(radius=0.05, color=color).move_to(UP*(1 + i/4) + position) for i in range(1, 4)])
            ellipsis_below = VGroup(*[Dot(radius=0.05, color=color).move_to(DOWN*(2 + i/4) + position) for i in range(1, 4)])
            layer.add(ellipsis_above, ellipsis_below)
            
        return layer

    def animate_layers(self, layers):
        for layer in layers:
            self.play(Create(layer))

        # connect the layers with edges
        for i in range(len(layers)-1):
            self.connect_layers(layers[i], layers[i+1], has_first_layer=(i==0))

    def connect_layers(self, layer1, layer2, has_first_layer=False, color=ManimColor("#000000")):
        # Draw lines between nodes of two layers
        lines = VGroup()
        if has_first_layer:
            # skip the last two nodes (=ellipsis dots) of the input layer
            start_layer = layer1[0:-2]
        else:
            start_layer = layer1
            
        for neuron1 in start_layer:
            for neuron2 in layer2:
                # Create a line between two neurons
                line = Line(neuron1.get_center(), neuron2.get_center(), stroke_width=1, color=color)
                lines.add(line)
        self.play(Create(lines))

            
    def feed_images_with_proper_dots(self, layers, layer_values):
        # Simulate feeding an image to the network by showing dots moving through the connections
        for i in range(len(layers)-1):
            self.animate_activations_without_fadeout(layers[i], layers[i+1], layer_values[i], layer_values[i+1])

        
    def animate_activations_without_fadeout(self, layer1, layer2, layer1_values, layer2_values):
        # Simulate activations between two layers with varying dot sizes, moving simultaneously without fadeout
        animations = []
        dots = []  # Store dots to fade them out later

        for neuron1, value1 in zip(layer1, layer1_values):
            for neuron2, value2 in zip(layer2, layer2_values):
                # Create a dot at the center of the connection
                dot = Dot(color=ManimColor("#ffdd04"))
                dot.move_to(neuron1.get_center())

                # Define different sizes for dots to simulate varying activations
                start_size = value1/4
                end_size = value2/4

                dot.scale(start_size)  # Set initial size of dot
                dots.append(dot)  # Store the dot for later fading

                # Create an animation for the dot moving between the two neurons and growing/shrinking in size
                move_dot = dot.animate.scale(end_size / start_size).move_to(neuron2.get_center())
                animations.append(move_dot)

        # Animate all activations simultaneously (without fading out)
        self.play(AnimationGroup(*animations, run_time=1))

        # Fade out all dots after the movement animation
        #self.play(FadeOut(*dots))     
        for dot in dots:
            self.remove(dot)
        
    def show_dots_at_layer(self, layer, values):
        # Show dots at the neurons of a layer with varying sizes based on the values
        dots = []
        for neuron, value in zip(layer, values):
            dot = Dot(color=ManimColor("#ffdd04"))
            dot.move_to(neuron.get_center())
            dot.scale(value/4)
            self.add(dot)
            dots.append(dot)    
        return dots

    def show_output_vector(self, output_layer, output_vector_values):
        # apply softmax to the output vector
        output_vector_values = np.exp(output_vector_values) / np.sum(np.exp(output_vector_values))
        # Normalize the output vector values to sum to 1
        output_vector_values = output_vector_values / np.sum(output_vector_values)
        # multiply by 100 and round to 1 decimal place
        output_vector_values = np.round(output_vector_values * 100, 1)
        # Simulate showing the classification result
        vector_string = r"y = \begin{bmatrix}"
        for value in output_vector_values[::-1]:
            vector_string += f"{value:.1f} \\\\ "
        vector_string += r"\end{bmatrix}"
        output_vector = MathTex(vector_string).scale(0.8)
        output_vector.set_color(BLACK)
        output_vector.to_edge(RIGHT)
        self.play(Write(output_vector))
        self.wait(2)
        self.play(FadeOut(output_vector))
        
    def add_labels(self, output_layer, labels):
        # Create labels for each output node (Book, Pen, Tape)
        label_texts = []
        for i, neuron in enumerate(output_layer):
            label = Text(labels[i], font="Roboto").scale(0.6)
            label_texts.append(label)
            label.next_to(neuron, RIGHT)  # Place label to the right of the neuron
            self.play(Write(label), run_time=0.5)
        return label_texts
    
    def create_image_stack(self, num_images=5, image_size=1.0, stack_offset=0.2):
        """
        Creates an abstract stack of images (squares or rectangles) positioned on the left side of the screen.
        
        Args:
        - num_images (int): Number of images to stack.
        - image_size (float): Size of each image (square).
        - stack_offset (float): Offset between consecutive images in the stack.
        
        Returns:
        - VGroup: A group of image rectangles stacked together.
        """
        # Create a VGroup to hold the stack of images
        image_stack = VGroup()
        
        # Loop to create and stack images
        for i in range(num_images):
            # Create an abstract image (square or rectangle)
            image = Square(side_length=image_size, fill_opacity=0.8, color=BLUE)
            image.set_fill(BLUE, opacity=0.8)  # Add some fill color

            # Offset each image slightly for the stacking effect
            image.shift(UP * i * stack_offset + LEFT * 6)

            # Add the image to the stack group
            image_stack.add(image)
            
        # Load and add an icon to the top image
        top_image = image_stack[-1]  # The top image in the stack
        icon = SVGMobject(".\\nn\\images\\image_stack.svg").scale(0.5)  # Load the icon and scale it
        icon.move_to(top_image.get_center())  # Center the icon on the top image
        image_stack.add(icon)
        
        # Return the stacked group of images
        return image_stack