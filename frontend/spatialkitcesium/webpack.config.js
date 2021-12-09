const path = require('path');

const webpack = require('webpack');
const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyWebpackPlugin = require("copy-webpack-plugin");

// The path to the CesiumJS source code
const cesiumSource = 'node_modules/cesium/Source';
const cesiumWorkers = '../Build/Cesium/Workers';

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
    sourcePrefix: ''
  },
  amd: {
      // Enable webpack-friendly use of require in Cesium
      toUrlUndefined: true
  },
  node: {
      // Resolve node module use of fs
      //fs: 'empty'
  },
  resolve: {
    alias: {
      cesium: path.resolve(__dirname, cesiumSource)
    },
    mainFiles: ['index', 'Cesium'],
    extensions: ['.ts', '.tsx', '.js'],
    fallback: {
      fs: false
    },
    modules: [
      path.join(__dirname, 'node_modules')
  ]
    //alias: {
      // CesiumJS module name
      //cesium: path.resolve(__dirname, cesiumSource)
    //}
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
    new HtmlWebpackPlugin({
      template: "./src/html/index.html",
    }),
    new CopyWebpackPlugin({
      patterns: [
        { from: path.join(cesiumSource, cesiumWorkers), to: 'Workers' },
        { from: path.join(cesiumSource, 'Assets'), to: 'Assets' },
        { from: path.join(cesiumSource, 'Widgets'), to: 'Widgets' }
    ]
      //patterns: [
      //  { from: "node_modules/cesium/Build/Cesium/Workers/", to: "Workers" },
      //  { from: "node_modules/cesium/Build/Cesium/ThirdParty/", to: "ThirdParty" },
      //  { from: "node_modules/cesium/Build/Cesium/Assets/", to: "Assets" },
      //  { from: "node_modules/cesium/Build/Cesium/Widgets/", to: "Widgets" },
      //],
    }),
    new webpack.DefinePlugin({
      CESIUM_BASE_URL: JSON.stringify(""),
    }),
  ]
};