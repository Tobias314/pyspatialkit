const path = require('path');

module.exports = {
  mode: 'development',
  context: path.resolve(__dirname, 'src'),
  entry: './app/app.tsx',
  devtool: 'source-map',
  devServer: {
    static: './public',
  },
  output: {
    filename: 'build/bundle.js',
    path: path.resolve(__dirname, 'public'),
  },
  resolve: {
    extensions: ['.ts', '.tsx', '.js']
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: ['ts-loader'],
      }
    ]
  }
};