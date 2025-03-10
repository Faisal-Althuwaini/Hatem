import React from "react";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "./ui/button";
import { NavLink } from "react-router-dom";

export default function CardItem({ title, content, url, target }) {
  return (
    <div className="p-4 max-w-sm sm:max-w-md md:max-w-xl md:mx-auto">
      <Card className="bg-cyan-500 text-white shadow-lg rounded-xl transition-transform duration-300 hover:scale-105 h-auto">
        <CardHeader>
          <CardTitle className="text-xl sm:text-2xl font-bold drop-shadow-md pt-2">
            {title}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-white/90 text-sm sm:text-base  leading-relaxed">
            {content}
          </p>
        </CardContent>
        <CardFooter className="flex justify-start items-end pt-2">
          <NavLink to={url} target={target}>
            <Button className="bg-white cursor-pointer text-cyan-600 hover:bg-blue-100 font-semibold px-4 pt-2 rounded-lg shadow-md transition-all text-sm sm:text-base">
              اضغط هنا
            </Button>
          </NavLink>
        </CardFooter>
      </Card>
    </div>
  );
}
