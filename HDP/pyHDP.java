import py4j.GatewayServer;

import java.util.Arrays;

import hdp.ProbabilityTree;

public class pyHDP {
  public static int[][] createFromPy4j(byte[] data) {
    java.nio.ByteBuffer buf = java.nio.ByteBuffer.wrap(data);
    int n = buf.getInt(), m = buf.getInt();
    int[][] matrix = new int[n][m];
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j)
        matrix[i][j] = buf.getInt();
    return matrix;
  }

  public static ProbabilityTree train_hdp(int[][] data) {
    ProbabilityTree tree = new ProbabilityTree();
    tree.addDataset(data);
    return tree;
  }

  public String getGreeting() {
    return "Hello, World!";
  }

  public static void main(String[] args) {
    pyHDP pyhdp = new pyHDP();
    // GatewayServer serves the pyHDP object to the Python side
    GatewayServer server = new GatewayServer(pyhdp);
    server.start();
    System.out.println("Gateway Server Started");
  }
}
