import React, { useState } from "react";
import {
  View,
  Text,
  Switch,
  Picker,
  StyleSheet,
  Button,
  ScrollView,
} from "react-native";

const SettingsPage = () => {
  // State for the settings
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState("en");

  // Toggles
  const toggleDarkMode = () => setIsDarkMode(!isDarkMode);
  const toggleNotifications = () =>
    setNotificationsEnabled(!notificationsEnabled);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      {/* Theme Setting */}
      <View style={styles.settingItem}>
        <Text style={styles.settingText}>Dark Mode</Text>
        <Switch value={isDarkMode} onValueChange={toggleDarkMode} />
      </View>

      {/* Notifications Setting */}
      <View style={styles.settingItem}>
        <Text style={styles.settingText}>Enable Notifications</Text>
        <Switch
          value={notificationsEnabled}
          onValueChange={toggleNotifications}
        />
      </View>

      {/* Language Selection */}
      <View style={styles.settingItem}>
        <Text style={styles.settingText}>Language</Text>
        <Picker
          selectedValue={selectedLanguage}
          style={styles.picker}
          onValueChange={(itemValue) => setSelectedLanguage(itemValue)}
        >
          <Picker.Item label="English" value="en" />
          <Picker.Item label="Japanese" value="jp" />
          <Picker.Item label="French" value="fr" />
        </Picker>
      </View>

      {/* Profile Management Button */}
      <View style={styles.settingItem}>
        <Button
          title="Manage Profile"
          onPress={() => alert("Profile management pressed")}
        />
      </View>

      {/* Logout Button */}
      <View style={styles.settingItem}>
        <Button
          title="Logout"
          color="red"
          onPress={() => alert("Logged out")}
        />
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 20,
  },
  settingItem: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginVertical: 15,
  },
  settingText: {
    fontSize: 18,
  },
  picker: {
    height: 50,
    width: 150,
  },
});

export default SettingsPage;
