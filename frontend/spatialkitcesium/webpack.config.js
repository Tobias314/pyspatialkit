const path = require('path');

const webpack = require('webpack');
const HtmlPlugin = require("html-webpack-plugin");
const CopyWebpackPlugin = require("copy-webpack-plugin");

module.exports = {
  mode: 'development',
  //context: path.resolve(__dirname, 'src'),
  entry: './src/app/app.tsx',
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
      },
      {
      test: /\.css$/,
      use: [ 'style-loader', 'css-loader' ]
      },
      {
      test: /\.(png|gif|jpg|jpeg|svg|xml|json)$/,
      use: [ 'url-loader' ]
      }
    ]
  },
  plugins: [
    new HtmlPlugin({
      template: "./src/html/index.html",
    }),
    new CopyWebpackPlugin({
      patterns: [
        { from: "node_modules/cesium/Build/Cesium/Workers/", to: "Workers" },
        { from: "node_modules/cesium/Build/Cesium/ThirdParty/", to: "ThirdParty" },
        { from: "node_modules/cesium/Build/Cesium/Assets/", to: "Assets" },
        { from: "node_modules/cesium/Build/Cesium/Widgets/", to: "Widgets" },
      ],
    }),
    new webpack.DefinePlugin({
      CESIUM_BASE_URL: JSON.stringify(""),
    }),
  ]
};