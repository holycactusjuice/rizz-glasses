import React, { useState } from "react";
import {
  View,
  TouchableOpacity,
  StyleSheet,
  Alert,
  TextInput,
} from "react-native";
import { MaterialIcons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import axios from "axios";

const HomeScreen = () => {
  const [text, setText] = useState("");
  const piAddress = "http://172.20.10.8:6000"; // Replace with your Raspberry Pi's IP

  const startRecording = async () => {
    try {
      const response = await axios.post(`${piAddress}/start-recording`, {
        context: text,
      });
      Alert.alert("Success", response.data);
    } catch (error) {
      Alert.alert("Error", "Could not start recording");
      console.error(error);
    }
  };

  const stopRecording = async () => {
    try {
      const response = await axios.get(`${piAddress}/stop-recording`);
      Alert.alert("Success", response.data);
    } catch (error) {
      Alert.alert("Error", "Could not stop recording");
      console.error(error);
    }
  };

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.textField}
        placeholder="Enter your context here"
        placeholderTextColor="gray"
        value={text}
        onChangeText={setText}
        multiline
        returnKeyType="done"
        blurOnSubmit={true}
      />
      <View style={styles.buttonContainer}>
        {/* Record Button with reversed gradient */}
        <TouchableOpacity onPress={startRecording}>
          <LinearGradient
            // Gradient starting from the center (inside) to the outside
            start={{ x: 0.5, y: 0.5 }} // Middle of the button
            end={{ x: 1, y: 1 }} // Toward the edges
            colors={["#ffffff", "#d3d3d3"]} // White at the center, gray at the edges
            style={styles.roundButton}
          >
            <MaterialIcons
              name="fiber-manual-record"
              size={50}
              color="tomato"
            />
          </LinearGradient>
        </TouchableOpacity>

        {/* Stop Button with reversed gradient */}
        <TouchableOpacity onPress={stopRecording}>
          <LinearGradient
            start={{ x: 0.5, y: 0.5 }} // Middle of the button
            end={{ x: 1, y: 1 }} // Toward the edges
            colors={["#ffffff", "#d3d3d3"]} // White at the center, gray at the edges
            style={styles.roundButton}
          >
            <MaterialIcons name="square" size={40} color="tomato" />
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 20,
  },
  textField: {
    width: "70%", // Takes up 70% of the screen width
    height: "40%", // Adjust height as needed
    borderColor: "gray",
    borderWidth: 1,
    borderRadius: 10,
    padding: 10,
    fontSize: 18,
    textAlign: "center", // Center the text
    marginBottom: 20, // Space between text field and buttons
  },
  buttonContainer: {
    flexDirection: "row", // Align buttons in a row
    justifyContent: "space-between",
    width: "55%", // Adjust to desired width
  },
  roundButton: {
    justifyContent: "center",
    alignItems: "center",
    width: 80,
    height: 80,
    borderRadius: 40, // Circular shape
    borderWidth: 2, // Outline thickness
    borderColor: "gray", // Outline color
    // Shadow properties for both iOS and Android
    shadowColor: "gray",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
    elevation: 8, // Required for Android shadow
  },
});

export default HomeScreen;
