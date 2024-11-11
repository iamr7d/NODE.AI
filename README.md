# NODE.AI


This web application is designed to create, explore, and analyze interactive story plots using a tree-like structure. Each node represents a situation in the story, and the user can select, modify, and explore future situations in the story plot. The application integrates with the Groq API to generate next-situation predictions based on user input.

## Features

- **Story Creation**: Start with an initial situation and genre, then generate possible next situations.
- **Interactive Story Tree**: Navigate through the plot by selecting or regenerating situations. Each situation is represented as a node in a tree.
- **Analytics**: Visualize the progress of the story with Plotly-powered analytics, including:
  - **Story Progress**: A graph showing the progression of the story by situation weight.
  - **Genre Distribution**: A pie chart displaying the distribution of genres in the story.
  - **Path Complexity**: A bar chart showing the complexity of story paths.
- **Unexplored Paths**: Identify and explore unexplored paths in the story tree.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.x
- Flask
- Plotly
- Groq API Client

### Install Dependencies

Clone the repository and install dependencies using `pip`:

```bash
git clone https://github.com/your-username/storytree-webapp.git
cd storytree-webapp
pip install -r requirements.txt

